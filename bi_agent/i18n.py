"""Language support: the five report languages, site-language detection, and report strings.

The model always returns stable English keys (JSON field names, enum values, the strategic
questions and maturity dimensions). Everything a reader sees in the report is looked up
here, so adding a language means adding one column to each table below.
"""

from __future__ import annotations

import re
from collections import Counter
from typing import Iterable

SUPPORTED: tuple[str, ...] = ("en", "pt-br", "fr", "de", "es")
DEFAULT_LANG = "en"

LANGUAGE_NAMES: dict[str, str] = {
    "en": "English",
    "pt-br": "Brazilian Portuguese",
    "fr": "French",
    "de": "German",
    "es": "Spanish",
}


def normalize_lang(tag: str | None) -> str | None:
    """Map a BCP-47-ish tag (``pt-BR``, ``pt_PT``, ``en-US``, ``de``) to a supported code, else None."""
    if not tag:
        return None
    primary = re.split(r"[-_]", tag.strip().lower(), maxsplit=1)[0]
    if primary == "pt":
        return "pt-br"
    return primary if primary in SUPPORTED else None


# Function words that are frequent and, within these five languages, distinctive. Words shared
# between languages (de, la, que, para, en ...) are left out so they cannot tip the vote.
_STOPWORDS: dict[str, frozenset[str]] = {
    "en": frozenset("the and of to with for our your is are that this we you from by".split()),
    "pt-br": frozenset("da do das dos com não uma você são nossa nosso também ao às pelo pela seu sua mais".split()),
    "fr": frozenset("le les du et pour avec une nous notre votre vous est sont dans sur au aux".split()),
    "de": frozenset("der die das und für mit ist sind wir ihre unsere nicht ein eine zu den auf sie".split()),
    "es": frozenset("el los del con una y nuestra nuestro también es al por sus más su".split()),
}
_WORD = re.compile(r"[^\W\d_]+", re.UNICODE)


def guess_text_lang(text: str, min_hits: int = 12) -> str | None:
    """Stop-word vote over the text. Returns a language only when the winner is clear."""
    counts: Counter[str] = Counter()
    for word in _WORD.findall(text.lower()):
        for lang, words in _STOPWORDS.items():
            if word in words:
                counts[lang] += 1
    if not counts:
        return None
    ranked = counts.most_common(2)
    best, hits = ranked[0]
    runner_up = ranked[1][1] if len(ranked) > 1 else 0
    if hits >= min_hits and hits >= 1.5 * runner_up:
        return best
    return None


def detect_site_lang(pages: Iterable[object]) -> str:
    """Language of a crawled site: page text first, then ``<html lang>``, then English.

    Text wins over the ``lang`` attribute because site templates often ship a default
    (``en-US``) that nobody changed while the content is in another language.
    """
    pages = list(pages)
    text = "\n".join(getattr(p, "text", "")[:5_000] for p in pages[:10])
    guessed = guess_text_lang(text)
    if guessed:
        return guessed
    votes: Counter[str] = Counter()
    for i, p in enumerate(pages):
        lang = normalize_lang(getattr(p, "lang", None))
        if lang:
            votes[lang] += 3 if i == 0 else 1  # the home page speaks for the site
    return votes.most_common(1)[0][0] if votes else DEFAULT_LANG


# --------------------------------------------------------------------------- report strings

# key: (en, pt-br, fr, de, es) — same order as SUPPORTED.
STRINGS: dict[str, tuple[str, str, str, str, str]] = {
    # classifications
    "verified_fact": ("Verified fact", "Fato verificado", "Fait vérifié", "Verifizierte Tatsache", "Hecho verificado"),
    "company_claim": ("Company claim", "Afirmação da empresa", "Affirmation de l'entreprise", "Angabe des Unternehmens", "Afirmación de la empresa"),
    "third_party_claim": ("Third-party claim", "Afirmação de terceiros", "Affirmation de tiers", "Angabe Dritter", "Afirmación de terceros"),
    "analytical_inference": ("Analytical inference", "Inferência analítica", "Inférence analytique", "Analytische Schlussfolgerung", "Inferencia analítica"),
    "unknown": ("Unknown", "Desconhecido", "Inconnu", "Unbekannt", "Desconocido"),
    # source types
    "first_party": ("first party", "fonte própria", "source propre", "Erstquelle", "fuente propia"),
    "third_party": ("third party", "terceiros", "tiers", "Drittquelle", "terceros"),
    # pain kinds
    "pain.functional": ("functional", "funcional", "fonctionnel", "funktional", "funcional"),
    "pain.financial": ("financial", "financeiro", "financier", "finanziell", "financiero"),
    "pain.operational": ("operational", "operacional", "opérationnel", "operativ", "operativo"),
    "pain.technical": ("technical", "técnico", "technique", "technisch", "técnico"),
    "pain.regulatory": ("regulatory", "regulatório", "réglementaire", "regulatorisch", "regulatorio"),
    "pain.strategic": ("strategic", "estratégico", "stratégique", "strategisch", "estratégico"),
    # competitor categories
    "cat.direct": ("direct", "direto", "direct", "direkt", "directo"),
    "cat.indirect": ("indirect", "indireto", "indirect", "indirekt", "indirecto"),
    "cat.incumbent": ("incumbent", "estabelecido", "acteur établi", "etablierter Anbieter", "establecido"),
    "cat.emerging": ("emerging", "emergente", "émergent", "aufstrebend", "emergente"),
    "cat.internal_alternative": ("internal alternative", "alternativa interna", "alternative interne", "interne Alternative", "alternativa interna"),
    # reproducibility
    "repro.easy": ("easy", "fácil", "facile", "leicht", "fácil"),
    "repro.moderate": ("moderate", "moderada", "modérée", "mittel", "moderada"),
    "repro.hard": ("hard", "difícil", "difficile", "schwer", "difícil"),
    "repro.unknown": ("unknown", "desconhecida", "inconnue", "unbekannt", "desconocida"),
    # source kinds (tiers of the research brief)
    "source_kind": ("Source kind", "Natureza da fonte", "Nature de la source", "Quellenart", "Naturaleza de la fuente"),
    "src.government_regulatory": ("government / regulator", "governo / regulador", "administration / régulateur", "Behörde / Regulierer", "gobierno / regulador"),
    "src.company_filing": ("statutory filing", "documento regulatório", "dépôt légal", "Pflichtveröffentlichung", "documento regulatorio"),
    "src.official_company": ("company material", "material da empresa", "document de l'entreprise", "Unternehmensangabe", "material de la empresa"),
    "src.investor_disclosure": ("investor disclosure", "informação a investidores", "communication aux investisseurs", "Investoreninformation", "información a inversores"),
    "src.partner_customer": ("partner / customer", "parceiro / cliente", "partenaire / client", "Partner / Kunde", "socio / cliente"),
    "src.industry_publication": ("industry publication", "publicação setorial", "publication sectorielle", "Fachpublikation", "publicación sectorial"),
    "src.news": ("news", "imprensa", "presse", "Presse", "prensa"),
    "src.database_aggregator": ("database / aggregator", "base de dados / agregador", "base de données / agrégateur", "Datenbank / Aggregator", "base de datos / agregador"),
    "src.forum_social": ("forum / social media", "fórum / rede social", "forum / réseau social", "Forum / soziale Medien", "foro / red social"),
    "website_subject": ("Website represents", "O site representa", "Le site représente", "Die Website stellt dar", "El sitio representa"),
    "thin_site_note": ("The website yielded little crawlable text ({chars} characters; {js} pages that only render with JavaScript), so this report relies mostly on external sources.",
                       "O site forneceu pouco texto rastreável ({chars} caracteres; {js} páginas que só aparecem com JavaScript), por isso este relatório se apoia principalmente em fontes externas.",
                       "Le site a fourni peu de texte exploitable ({chars} caractères ; {js} pages qui ne s'affichent qu'avec JavaScript) ; ce rapport repose donc surtout sur des sources externes.",
                       "Die Website lieferte wenig auswertbaren Text ({chars} Zeichen; {js} Seiten, die nur mit JavaScript erscheinen); dieser Bericht stützt sich daher vor allem auf externe Quellen.",
                       "El sitio ofreció poco texto rastreable ({chars} caracteres; {js} páginas que solo se muestran con JavaScript), por lo que este informe se apoya sobre todo en fuentes externas."),
    # offering kinds
    "kind.core_product": ("core product", "produto principal", "produit principal", "Kernprodukt", "producto principal"),
    "kind.secondary_product": ("secondary product", "produto secundário", "produit secondaire", "Nebenprodukt", "producto secundario"),
    "kind.service": ("service", "serviço", "service", "Dienstleistung", "servicio"),
    "kind.professional_services": ("professional services", "serviços profissionais", "services professionnels", "professionelle Dienstleistungen", "servicios profesionales"),
    "kind.subscription": ("subscription", "assinatura", "abonnement", "Abonnement", "suscripción"),
    "kind.platform": ("platform", "plataforma", "plateforme", "Plattform", "plataforma"),
    "kind.api": ("API", "API", "API", "API", "API"),
    "kind.software": ("software", "software", "logiciel", "Software", "software"),
    "kind.hardware": ("hardware", "hardware", "matériel", "Hardware", "hardware"),
    "kind.data_product": ("data product", "produto de dados", "produit de données", "Datenprodukt", "producto de datos"),
    "kind.marketplace": ("marketplace", "marketplace", "place de marché", "Marktplatz", "marketplace"),
    "kind.licensing": ("licensing", "licenciamento", "licence", "Lizenzierung", "licencias"),
    "kind.other": ("other", "outro", "autre", "sonstiges", "otro"),
    "strategic_note": ("These answers are analytical judgments drawn from the evidence cited, not facts stated by the company or by third parties.",
                       "Estas respostas são julgamentos analíticos baseados nas evidências citadas, não fatos declarados pela empresa ou por terceiros.",
                       "Ces réponses sont des jugements analytiques fondés sur les preuves citées, et non des faits énoncés par l'entreprise ou par des tiers.",
                       "Diese Antworten sind analytische Einschätzungen auf Grundlage der zitierten Belege, keine Tatsachen, die das Unternehmen oder Dritte angeben.",
                       "Estas respuestas son juicios analíticos basados en la evidencia citada, no hechos declarados por la empresa ni por terceros."),
    # header block
    "title": ("Company Intelligence Report: {name}", "Relatório de Inteligência Empresarial: {name}",
              "Rapport d'intelligence d'entreprise : {name}", "Unternehmensanalyse: {name}",
              "Informe de inteligencia empresarial: {name}"),
    "the_company": ("the company", "a empresa", "l'entreprise", "das Unternehmen", "la empresa"),
    "subject_url": ("Subject URL", "URL analisada", "URL analysée", "Analysierte URL", "URL analizada"),
    "generated": ("Report generated", "Relatório gerado em", "Rapport généré le", "Bericht erstellt am", "Informe generado el"),
    "evidence_items": ("Evidence items: {n} (website pages crawled: {pages})",
                       "Itens de evidência: {n} (páginas do site rastreadas: {pages})",
                       "Éléments de preuve : {n} (pages du site explorées : {pages})",
                       "Belege: {n} (gecrawlte Seiten der Website: {pages})",
                       "Elementos de evidencia: {n} (páginas del sitio rastreadas: {pages})"),
    "key": ("Classification key: every claim carries one of {labels}. Bracketed ids such as [E012] resolve to the Sources section.",
            "Legenda de classificação: toda afirmação recebe uma das categorias {labels}. Identificadores entre colchetes, como [E012], remetem à seção Fontes.",
            "Clé de classification : chaque affirmation porte l'une des mentions {labels}. Les identifiants entre crochets, comme [E012], renvoient à la section Sources.",
            "Klassifizierung: Jede Aussage trägt eine der Kennzeichnungen {labels}. Kennungen in eckigen Klammern wie [E012] verweisen auf den Abschnitt Quellen.",
            "Clave de clasificación: cada afirmación lleva una de las etiquetas {labels}. Los identificadores entre corchetes, como [E012], remiten a la sección Fuentes."),
    "or": ("or", "ou", "ou", "oder", "o"),
    # section titles
    "s1": ("Executive Summary", "Sumário Executivo", "Synthèse", "Zusammenfassung", "Resumen ejecutivo"),
    "s2": ("Company Snapshot", "Visão Geral da Empresa", "Fiche d'identité", "Unternehmensprofil", "Ficha de la empresa"),
    "s3": ("What the Company Does", "O Que a Empresa Faz", "Activité de l'entreprise", "Was das Unternehmen macht", "Qué hace la empresa"),
    "s4": ("Problems It Solves", "Problemas que Resolve", "Problèmes résolus", "Gelöste Probleme", "Problemas que resuelve"),
    "s5": ("Products and Services", "Produtos e Serviços", "Produits et services", "Produkte und Dienstleistungen", "Productos y servicios"),
    "s6": ("Customer Segments and Use Cases", "Segmentos de Clientes e Casos de Uso", "Segments de clientèle et cas d'usage", "Kundensegmente und Anwendungsfälle", "Segmentos de clientes y casos de uso"),
    "s7": ("Business Model and Monetization", "Modelo de Negócio e Monetização", "Modèle économique et monétisation", "Geschäftsmodell und Monetarisierung", "Modelo de negocio y monetización"),
    "s8": ("Go-to-Market Strategy", "Estratégia de Entrada no Mercado", "Stratégie de mise sur le marché", "Go-to-Market-Strategie", "Estrategia de salida al mercado"),
    "s9": ("Technology and Intellectual Property", "Tecnologia e Propriedade Intelectual", "Technologie et propriété intellectuelle", "Technologie und geistiges Eigentum", "Tecnología y propiedad intelectual"),
    "s10": ("Market Landscape", "Panorama de Mercado", "Paysage du marché", "Marktumfeld", "Panorama del mercado"),
    "s11": ("Competitive Landscape", "Panorama Competitivo", "Paysage concurrentiel", "Wettbewerbsumfeld", "Panorama competitivo"),
    "s12": ("Differentiation and Defensibility", "Diferenciação e Defensabilidade", "Différenciation et défendabilité", "Differenzierung und Verteidigungsfähigkeit", "Diferenciación y defendibilidad"),
    "s13": ("Customers, Partnerships and Ecosystem", "Clientes, Parcerias e Ecossistema", "Clients, partenariats et écosystème", "Kunden, Partnerschaften und Ökosystem", "Clientes, alianzas y ecosistema"),
    "s14": ("Financial and Funding Information", "Informações Financeiras e de Captação", "Informations financières et levées de fonds", "Finanz- und Finanzierungsinformationen", "Información financiera y de financiación"),
    "s15": ("Growth and Traction Signals", "Sinais de Crescimento e Tração", "Signaux de croissance et de traction", "Wachstums- und Traktionssignale", "Señales de crecimiento y tracción"),
    "s16": ("SWOT", "Análise SWOT", "Analyse SWOT", "SWOT-Analyse", "Análisis DAFO"),
    "s17": ("Risks and Red Flags", "Riscos e Sinais de Alerta", "Risques et signaux d'alerte", "Risiken und Warnsignale", "Riesgos y señales de alerta"),
    "s18": ("Strategic Opportunities", "Oportunidades Estratégicas", "Opportunités stratégiques", "Strategische Chancen", "Oportunidades estratégicas"),
    "s19": ("Analyst Observations", "Observações do Analista", "Observations de l'analyste", "Beobachtungen des Analysten", "Observaciones del analista"),
    "s20": ("Open Questions", "Questões em Aberto", "Questions ouvertes", "Offene Fragen", "Preguntas abiertas"),
    "s21": ("Sources", "Fontes", "Sources", "Quellen", "Fuentes"),
    # PDF page chrome
    "key_facts": ("Key facts", "Fatos principais", "Faits essentiels", "Eckdaten", "Datos clave"),
    "evidence_profile": ("Evidence profile: how the {n} labelled claims are grounded",
                         "Perfil das evidências: como se sustentam as {n} afirmações classificadas",
                         "Profil des preuves : fondement des {n} affirmations classées",
                         "Belegprofil: Grundlage der {n} gekennzeichneten Aussagen",
                         "Perfil de la evidencia: en qué se basan las {n} afirmaciones clasificadas"),
    "accessed_all": ("All sources accessed on {date}.", "Todas as fontes acessadas em {date}.",
                     "Toutes les sources ont été consultées le {date}.", "Alle Quellen abgerufen am {date}.",
                     "Todas las fuentes se consultaron el {date}."),
    "contents": ("Contents", "Sumário", "Sommaire", "Inhalt", "Índice"),
    "page_of": ("Page {n} of {total}", "Página {n} de {total}", "Page {n} sur {total}", "Seite {n} von {total}",
                "Página {n} de {total}"),
    # snapshot rows
    "field": ("Field", "Campo", "Champ", "Feld", "Campo"),
    "value": ("Value", "Valor", "Valeur", "Wert", "Valor"),
    "company": ("Company", "Empresa", "Entreprise", "Unternehmen", "Empresa"),
    "legal_entity": ("Legal entity", "Razão social", "Entité juridique", "Rechtsträger", "Razón social"),
    "website": ("Website", "Site", "Site web", "Website", "Sitio web"),
    "headquarters": ("Headquarters", "Sede", "Siège", "Hauptsitz", "Sede"),
    "founded": ("Founded", "Fundação", "Création", "Gegründet", "Fundación"),
    "founders": ("Founders", "Fundadores", "Fondateurs", "Gründer", "Fundadores"),
    "ownership": ("Ownership", "Controle acionário", "Actionnariat", "Eigentümerstruktur", "Estructura accionarial"),
    "public_private": ("Public / private", "Capital aberto / fechado", "Coté / non coté", "Börsennotiert / privat", "Cotizada / privada"),
    "ticker": ("Ticker", "Código de negociação", "Symbole boursier", "Börsenkürzel", "Símbolo bursátil"),
    "parent": ("Parent company", "Controladora", "Société mère", "Muttergesellschaft", "Empresa matriz"),
    "subsidiaries": ("Subsidiaries", "Subsidiárias", "Filiales", "Tochtergesellschaften", "Filiales"),
    "brands": ("Brands", "Marcas", "Marques", "Marken", "Marcas"),
    "leadership": ("Leadership", "Liderança", "Direction", "Führung", "Dirección"),
    "industry": ("Industry", "Setor", "Secteur", "Branche", "Sector"),
    "adjacent_industries": ("Adjacent industries", "Setores adjacentes", "Secteurs adjacents", "Angrenzende Branchen", "Sectores adyacentes"),
    "core_market": ("Core market", "Mercado principal", "Marché principal", "Kernmarkt", "Mercado principal"),
    "business_model": ("Business model", "Modelo de negócio", "Modèle économique", "Geschäftsmodell", "Modelo de negocio"),
    "customer_type": ("Customer type", "Tipo de cliente", "Type de client", "Kundentyp", "Tipo de cliente"),
    "geo_presence": ("Geographic presence", "Presença geográfica", "Présence géographique", "Geografische Präsenz", "Presencia geográfica"),
    "identity_uncertainties": ("Identity uncertainties", "Incertezas sobre a identidade", "Incertitudes sur l'identité", "Unsicherheiten zur Identität", "Incertidumbres sobre la identidad"),
    # tables and sub-headings
    "pain_type": ("Pain type", "Tipo de dor", "Type de problème", "Art des Problems", "Tipo de problema"),
    "problem": ("Problem", "Problema", "Problème", "Problem", "Problema"),
    "consequence": ("Consequence if unsolved", "Consequência se não resolvido", "Conséquence si non résolu", "Folge, wenn ungelöst", "Consecuencia si no se resuelve"),
    "evidence": ("Evidence", "Evidências", "Preuves", "Belege", "Evidencia"),
    "offering": ("Offering", "Oferta", "Offre", "Angebot", "Oferta"),
    "target_customer": ("Target Customer", "Cliente-alvo", "Client cible", "Zielkunde", "Cliente objetivo"),
    "problem_solved": ("Problem Solved", "Problema resolvido", "Problème résolu", "Gelöstes Problem", "Problema resuelto"),
    "capabilities": ("Key Capabilities", "Principais capacidades", "Capacités clés", "Kernfähigkeiten", "Capacidades clave"),
    "benefit": ("Business Benefit", "Benefício para o negócio", "Bénéfice métier", "Geschäftsnutzen", "Beneficio para el negocio"),
    "monetization": ("Monetization", "Monetização", "Monétisation", "Monetarisierung", "Monetización"),
    "basis": ("Basis", "Base", "Fondement", "Grundlage", "Base"),
    "no_offerings": ("No offerings could be extracted from the website", "Nenhuma oferta pôde ser extraída do site",
                     "Aucune offre n'a pu être extraite du site", "Aus der Website ließen sich keine Angebote ableiten",
                     "No se pudo extraer ninguna oferta del sitio"),
    "segments": ("Segments", "Segmentos", "Segments", "Segmente", "Segmentos"),
    "target_industries": ("Target industries", "Setores-alvo", "Secteurs cibles", "Zielbranchen", "Sectores objetivo"),
    "use_cases": ("Use cases", "Casos de uso", "Cas d'usage", "Anwendungsfälle", "Casos de uso"),
    "icp": ("Ideal customer profile", "Perfil de cliente ideal", "Profil de client idéal", "Ideales Kundenprofil", "Perfil de cliente ideal"),
    "buyer_user": ("Buyer, user and economic decision maker", "Comprador, usuário e decisor econômico",
                   "Acheteur, utilisateur et décideur économique", "Käufer, Nutzer und wirtschaftlicher Entscheider",
                   "Comprador, usuario y decisor económico"),
    "revenue_model": ("Revenue model", "Modelo de receita", "Modèle de revenus", "Erlösmodell", "Modelo de ingresos"),
    "pricing_signals": ("Pricing signals from the website", "Sinais de preço no site", "Indications tarifaires sur le site", "Preissignale auf der Website", "Señales de precios en el sitio"),
    "no_pricing": ("No pricing information published.", "Nenhuma informação de preço publicada.", "Aucune information tarifaire publiée.", "Keine Preisangaben veröffentlicht.", "No se publica información de precios."),
    "sales_signals": ("Sales and distribution signals from the website", "Sinais de vendas e distribuição no site",
                      "Signaux de vente et de distribution sur le site", "Vertriebssignale auf der Website",
                      "Señales de ventas y distribución en el sitio"),
    "tech_signals": ("Observed technology signals", "Sinais de tecnologia observados", "Signaux technologiques observés", "Beobachtete Technologiesignale", "Señales tecnológicas observadas"),
    "ip_claims": ("IP, regulatory and certification claims", "Afirmações sobre PI, regulação e certificações",
                  "Affirmations sur la PI, la réglementation et les certifications", "Angaben zu IP, Regulierung und Zertifizierungen",
                  "Afirmaciones sobre PI, regulación y certificaciones"),
    "primary_market": ("Primary market", "Mercado principal", "Marché principal", "Hauptmarkt", "Mercado principal"),
    "adjacent_markets": ("Adjacent markets", "Mercados adjacentes", "Marchés adjacents", "Angrenzende Märkte", "Mercados adyacentes"),
    "market_maturity": ("Market maturity", "Maturidade do mercado", "Maturité du marché", "Marktreife", "Madurez del mercado"),
    "trends": ("Structural trends", "Tendências estruturais", "Tendances structurelles", "Strukturelle Trends", "Tendencias estructurales"),
    "tech_shifts": ("Technological shifts", "Mudanças tecnológicas", "Évolutions technologiques", "Technologische Verschiebungen", "Cambios tecnológicos"),
    "regulatory": ("Regulatory influences", "Influências regulatórias", "Influences réglementaires", "Regulatorische Einflüsse", "Influencias regulatorias"),
    "behaviour": ("Customer behaviour changes", "Mudanças no comportamento do cliente", "Évolution des comportements clients", "Veränderungen im Kundenverhalten", "Cambios en el comportamiento del cliente"),
    "barriers": ("Barriers to entry", "Barreiras de entrada", "Barrières à l'entrée", "Markteintrittsbarrieren", "Barreras de entrada"),
    "switching": ("Switching costs", "Custos de troca", "Coûts de changement", "Wechselkosten", "Costes de cambio"),
    "commoditization": ("Commoditization risk", "Risco de comoditização", "Risque de banalisation", "Kommoditisierungsrisiko", "Riesgo de comoditización"),
    "consolidation": ("Consolidation dynamics", "Dinâmica de consolidação", "Dynamique de consolidation", "Konsolidierungsdynamik", "Dinámica de consolidación"),
    "sizing": ("Market sizing", "Dimensionamento de mercado", "Taille du marché", "Marktgröße", "Tamaño del mercado"),
    "metric": ("Metric", "Métrica", "Indicateur", "Kennzahl", "Métrica"),
    "year": ("Year", "Ano", "Année", "Jahr", "Año"),
    "methodology": ("Methodology", "Metodologia", "Méthodologie", "Methodik", "Metodología"),
    "limitations": ("Limitations", "Limitações", "Limites", "Einschränkungen", "Limitaciones"),
    "source": ("Source", "Fonte", "Source", "Quelle", "Fuente"),
    "no_sizing": ("No credible sourced market-size figure was found; none is estimated here.",
                  "Não foi encontrado nenhum número de tamanho de mercado com fonte confiável; nenhum é estimado aqui.",
                  "Aucun chiffre de taille de marché crédible et sourcé n'a été trouvé ; aucun n'est estimé ici.",
                  "Es wurde keine belastbare, belegte Marktgröße gefunden; hier wird keine geschätzt.",
                  "No se encontró ninguna cifra creíble y con fuente sobre el tamaño del mercado; aquí no se estima ninguna."),
    "category": ("Category", "Categoria", "Catégorie", "Kategorie", "Categoría"),
    "target_segment": ("Target Segment", "Segmento-alvo", "Segment cible", "Zielsegment", "Segmento objetivo"),
    "strength": ("Key Strength", "Principal força", "Force principale", "Hauptstärke", "Fortaleza principal"),
    "difference": ("Key Difference", "Principal diferença", "Différence principale", "Hauptunterschied", "Diferencia principal"),
    "dimension": ("Dimension", "Dimensão", "Dimension", "Dimension", "Dimensión"),
    "claimed_diff": ("Claimed differentiation", "Diferenciação declarada", "Différenciation revendiquée", "Behauptete Differenzierung", "Diferenciación declarada"),
    "observable_diff": ("Observable differentiation", "Diferenciação observável", "Différenciation observable", "Beobachtbare Differenzierung", "Diferenciación observable"),
    "reproducibility": ("Reproducibility", "Reprodutibilidade", "Reproductibilité", "Nachahmbarkeit", "Reproducibilidad"),
    "named_customers": ("Named customers and case studies", "Clientes citados e estudos de caso", "Clients cités et études de cas", "Genannte Kunden und Fallstudien", "Clientes citados y casos de estudio"),
    "no_customers": ("No named customers found.", "Nenhum cliente citado encontrado.", "Aucun client cité trouvé.", "Keine namentlich genannten Kunden gefunden.", "No se encontraron clientes citados."),
    "partnerships": ("Partnerships and integrations", "Parcerias e integrações", "Partenariats et intégrations", "Partnerschaften und Integrationen", "Alianzas e integraciones"),
    "no_partnerships": ("No partnerships or integrations found.", "Nenhuma parceria ou integração encontrada.", "Aucun partenariat ni intégration trouvé.", "Keine Partnerschaften oder Integrationen gefunden.", "No se encontraron alianzas ni integraciones."),
    "geography": ("Geography", "Geografia", "Géographie", "Geografie", "Geografía"),
    "no_financials": ("No financial or funding information is publicly available; nothing is estimated here.",
                      "Não há informações financeiras ou de captação disponíveis publicamente; nada é estimado aqui.",
                      "Aucune information financière ou de financement n'est publique ; rien n'est estimé ici.",
                      "Es sind keine Finanz- oder Finanzierungsinformationen öffentlich verfügbar; hier wird nichts geschätzt.",
                      "No hay información financiera ni de financiación pública; aquí no se estima nada."),
    "commercial_signals": ("Commercial signals", "Sinais comerciais", "Signaux commerciaux", "Kommerzielle Signale", "Señales comerciales"),
    "organization": ("Organization and talent", "Organização e talentos", "Organisation et talents", "Organisation und Talente", "Organización y talento"),
    "hiring": ("Hiring signals from the careers pages", "Sinais de contratação nas páginas de carreiras",
               "Signaux de recrutement sur les pages carrières", "Einstellungssignale auf den Karriereseiten",
               "Señales de contratación en las páginas de empleo"),
    "no_careers": ("No careers page content found.", "Nenhum conteúdo de página de carreiras encontrado.", "Aucun contenu de page carrières trouvé.", "Keine Inhalte von Karriereseiten gefunden.", "No se encontró contenido de páginas de empleo."),
    "strengths": ("Strengths", "Forças", "Forces", "Stärken", "Fortalezas"),
    "weaknesses": ("Weaknesses", "Fraquezas", "Faiblesses", "Schwächen", "Debilidades"),
    "opportunities": ("Opportunities", "Oportunidades", "Opportunités", "Chancen", "Oportunidades"),
    "threats": ("Threats", "Ameaças", "Menaces", "Risiken", "Amenazas"),
    "strategic_analysis": ("Strategic analysis", "Análise estratégica", "Analyse stratégique", "Strategische Analyse", "Análisis estratégico"),
    "business_maturity": ("Business maturity", "Maturidade do negócio", "Maturité de l'entreprise", "Unternehmensreife", "Madurez del negocio"),
    "no_red_flags": ("No red flags were identified in the evidence gathered; absence of evidence is not evidence of absence.",
                     "Nenhum sinal de alerta foi identificado nas evidências coletadas; ausência de evidência não é evidência de ausência.",
                     "Aucun signal d'alerte n'a été identifié dans les preuves recueillies ; l'absence de preuve n'est pas la preuve de l'absence.",
                     "In den gesammelten Belegen wurden keine Warnsignale gefunden; fehlende Belege sind kein Beleg für deren Fehlen.",
                     "No se identificaron señales de alerta en la evidencia reunida; la ausencia de evidencia no es evidencia de ausencia."),
    "type": ("Type", "Tipo", "Type", "Art", "Tipo"),
    "opportunity": ("Opportunity", "Oportunidade", "Opportunité", "Chance", "Oportunidad"),
    "why_exists": ("Why it exists", "Por que existe", "Pourquoi elle existe", "Warum sie besteht", "Por qué existe"),
    "research_incomplete": ("Could not complete research (unfinished groups):", "Não foi possível concluir a pesquisa (grupos pendentes):", "Recherche inachevée (groupes restants) :", "Recherche nicht abgeschlossen (offene Gruppen):", "No se pudo completar la investigación (grupos pendientes):"),
    "research_gap": ("Research gap: {gap}", "Lacuna de pesquisa: {gap}", "Lacune de recherche : {gap}", "Recherchelücke: {gap}", "Laguna de investigación: {gap}"),
    "none": ("None.", "Nenhuma.", "Aucune.", "Keine.", "Ninguna."),
    "id": ("Id", "Id", "Id", "Id", "Id"),
    "title_col": ("Title", "Título", "Titre", "Titel", "Título"),
    "publisher": ("Publisher", "Publicado por", "Éditeur", "Herausgeber", "Editor"),
    "url": ("URL", "URL", "URL", "URL", "URL"),
    "published": ("Published", "Publicação", "Publication", "Veröffentlicht", "Publicación"),
    "accessed": ("Accessed", "Acesso em", "Consulté le", "Abgerufen", "Consultado"),
    "n_a": ("n/a", "n/d", "n.d.", "k. A.", "n/d"),
    "discarded": ("**Discarded during verification** (sources the research model cited that were never returned by search):",
                  "**Descartado na verificação** (fontes citadas pelo modelo de pesquisa que nunca foram retornadas pela busca):",
                  "**Écarté lors de la vérification** (sources citées par le modèle de recherche qui n'ont jamais été renvoyées par la recherche) :",
                  "**Bei der Prüfung verworfen** (Quellen, die das Recherchemodell zitiert hat, die aber nie von der Suche geliefert wurden):",
                  "**Descartado en la verificación** (fuentes citadas por el modelo de investigación que la búsqueda nunca devolvió):"),
    "nothing": ("Nothing established from the evidence gathered.", "Nada estabelecido a partir das evidências coletadas.",
                "Rien d'établi à partir des preuves recueillies.", "Aus den gesammelten Belegen ließ sich nichts feststellen.",
                "No se estableció nada a partir de la evidencia reunida."),
}

# English key (what the model returns) → (pt-br, fr, de, es)
STRATEGIC_QUESTION_TEXT: dict[str, tuple[str, str, str, str]] = {
    "What appears to be the company's strongest competitive asset?": (
        "Qual parece ser o ativo competitivo mais forte da empresa?",
        "Quel semble être l'atout concurrentiel le plus solide de l'entreprise ?",
        "Was scheint der stärkste Wettbewerbsvorteil des Unternehmens zu sein?",
        "¿Cuál parece ser el activo competitivo más sólido de la empresa?"),
    "What is easiest for competitors to replicate?": (
        "O que é mais fácil para os concorrentes replicarem?",
        "Qu'est-ce que les concurrents peuvent le plus facilement reproduire ?",
        "Was können Wettbewerber am leichtesten nachahmen?",
        "¿Qué es lo más fácil de replicar para los competidores?"),
    "What is hardest to replicate?": (
        "O que é mais difícil de replicar?",
        "Qu'est-ce qui est le plus difficile à reproduire ?",
        "Was ist am schwersten nachzuahmen?",
        "¿Qué es lo más difícil de replicar?"),
    "What could accelerate its growth?": (
        "O que poderia acelerar seu crescimento?",
        "Qu'est-ce qui pourrait accélérer sa croissance ?",
        "Was könnte das Wachstum beschleunigen?",
        "¿Qué podría acelerar su crecimiento?"),
    "What could constrain its growth?": (
        "O que poderia limitar seu crescimento?",
        "Qu'est-ce qui pourrait freiner sa croissance ?",
        "Was könnte das Wachstum bremsen?",
        "¿Qué podría limitar su crecimiento?"),
    "What could disrupt the company?": (
        "O que poderia provocar uma ruptura na empresa?",
        "Qu'est-ce qui pourrait bouleverser l'entreprise ?",
        "Was könnte das Unternehmen disruptiv bedrohen?",
        "¿Qué podría provocar una disrupción en la empresa?"),
    "What adjacent markets could it enter?": (
        "Em quais mercados adjacentes ela poderia entrar?",
        "Sur quels marchés adjacents pourrait-elle se développer ?",
        "In welche angrenzenden Märkte könnte es eintreten?",
        "¿En qué mercados adyacentes podría entrar?"),
    "What partnerships would make strategic sense?": (
        "Quais parcerias fariam sentido estratégico?",
        "Quels partenariats auraient un sens stratégique ?",
        "Welche Partnerschaften wären strategisch sinnvoll?",
        "¿Qué alianzas tendrían sentido estratégico?"),
    "What capabilities might it acquire rather than build?": (
        "Quais capacidades ela poderia adquirir em vez de desenvolver?",
        "Quelles capacités pourrait-elle acquérir plutôt que développer ?",
        "Welche Fähigkeiten könnte es zukaufen statt selbst aufzubauen?",
        "¿Qué capacidades podría adquirir en lugar de desarrollar?"),
    "What would materially increase or decrease the company's strategic value?": (
        "O que aumentaria ou reduziria significativamente o valor estratégico da empresa?",
        "Qu'est-ce qui augmenterait ou diminuerait sensiblement la valeur stratégique de l'entreprise ?",
        "Was würde den strategischen Wert des Unternehmens wesentlich steigern oder mindern?",
        "¿Qué aumentaría o reduciría de forma significativa el valor estratégico de la empresa?"),
}

MATURITY_DIMENSION_TEXT: dict[str, tuple[str, str, str, str]] = {
    "Product maturity": ("Maturidade do produto", "Maturité du produit", "Produktreife", "Madurez del producto"),
    "Commercial maturity": ("Maturidade comercial", "Maturité commerciale", "Kommerzielle Reife", "Madurez comercial"),
    "Market maturity": ("Maturidade de mercado", "Maturité du marché", "Marktreife", "Madurez de mercado"),
    "Technology maturity": ("Maturidade tecnológica", "Maturité technologique", "Technologische Reife", "Madurez tecnológica"),
    "Operational maturity": ("Maturidade operacional", "Maturité opérationnelle", "Operative Reife", "Madurez operativa"),
    "International maturity": ("Maturidade internacional", "Maturité internationale", "Internationale Reife", "Madurez internacional"),
    "Partner ecosystem": ("Ecossistema de parceiros", "Écosystème de partenaires", "Partner-Ökosystem", "Ecosistema de socios"),
    "Brand maturity": ("Maturidade da marca", "Maturité de la marque", "Markenreife", "Madurez de la marca"),
}


class Translator:
    """Report-string lookup for one language. Unknown keys fall back to the key itself."""

    def __init__(self, lang: str | None) -> None:
        self.lang = normalize_lang(lang) or DEFAULT_LANG
        self._i = SUPPORTED.index(self.lang)

    def __call__(self, key: str, **kwargs: object) -> str:
        row = STRINGS.get(key)
        text = row[self._i] if row else key
        return text.format(**kwargs) if kwargs else text

    def _keyed(self, table: dict[str, tuple[str, str, str, str]], key: str) -> str:
        if self._i == 0 or key not in table:
            return key
        return table[key][self._i - 1]

    def question(self, english: str) -> str:
        return self._keyed(STRATEGIC_QUESTION_TEXT, english)

    def dimension(self, english: str) -> str:
        return self._keyed(MATURITY_DIMENSION_TEXT, english)
