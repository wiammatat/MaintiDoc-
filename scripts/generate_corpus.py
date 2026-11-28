"""Génère le corpus synthétique MaintiDoc (maintenance SAN) dans data/.

Usage : python scripts/generate_corpus.py              # tout le corpus
        python scripts/generate_corpus.py --mcp-only   # seulement la base MCP (data/mcp/maintenance.db)
Toutes les données sont fictives. Le générateur est déterministe (graine fixe).
"""

import hashlib
import io
import json
import random
import sqlite3
import sys
from datetime import date, timedelta
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageEnhance, ImageFilter, ImageFont
from reportlab.lib import colors
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
from reportlab.lib.units import cm
from reportlab.platypus import Image as RLImage
from reportlab.platypus import PageBreak, Paragraph, SimpleDocTemplate, Spacer, Table, TableStyle

try:
    import pymupdf
except ImportError:
    import fitz as pymupdf

sys.path.insert(0, str(Path(__file__).parent))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from common.security import hash_password  # noqa: E402
from corpus_content import FAMILLES, GUIDE_MCP, MANUEL_SAN, NOTES, SCANS, SECURITE  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent / "data"
CORPUS = ROOT / "corpus"
FIGURES = ROOT / "corpus" / "figures"
PHOTOS = ROOT / "photos"
MCP = ROOT / "mcp"

random.seed(42)
np.random.seed(42)

STYLES = getSampleStyleSheet()
H1, H2, BODY = STYLES["Title"], STYLES["Heading2"], STYLES["BodyText"]
SMALL = ParagraphStyle("small", parent=BODY, fontSize=8.5, leading=10.5)
CAPTION = ParagraphStyle("caption", parent=BODY, fontSize=9, textColor=colors.grey, alignment=1)

COULEURS = {"vert": (40, 200, 70), "orange": (255, 150, 20), "rouge": (230, 30, 30), "bleu": (40, 120, 240)}
TOUS_LES_GROUPES = sorted({f["groupe"] for f in FAMILLES.values()} | {"responsable"})
LIBELLES_PLAQUE = {"capacite_brute": "Capacité", "alimentation": "Alimentation", "ports": "Ports"}

documents = []  # manifeste du corpus


# --------------------------------------------------------------------------- utilitaires

def font(size):
    for name in ("arial.ttf", "DejaVuSans.ttf", "LiberationSans-Regular.ttf"):
        try:
            return ImageFont.truetype(name, size)
        except OSError:
            continue
    return ImageFont.load_default(size=size)


def groupes(famille, restreint=False):
    if restreint:
        return ["responsable"]
    return [FAMILLES[famille]["groupe"], "responsable"]


def register(doc_id, titre, famille, type_doc, fichier, version="v1", modele=None, restreint=False, **extra):
    documents.append({
        "document_id": doc_id, "titre": titre, "version": version, "famille": famille,
        "modele": modele, "type": type_doc, "fichier": str(fichier.relative_to(ROOT)).replace("\\", "/"),
        "groupes_autorises": extra.pop("groupes", None) or groupes(famille, restreint), **extra,
    })


def table(rows, widths, header=True):
    data = [[Paragraph(str(c), SMALL) for c in r] for r in rows]
    t = Table(data, colWidths=widths, repeatRows=1 if header else 0)
    style = [("GRID", (0, 0), (-1, -1), 0.4, colors.grey), ("VALIGN", (0, 0), (-1, -1), "TOP")]
    if header:
        style.append(("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#dde4ee")))
    t.setStyle(TableStyle(style))
    return t


def build_pdf(path, doc_id, version, story):
    def footer(canvas, doc):
        canvas.saveState()
        canvas.setFont("Helvetica", 8)
        canvas.setFillColor(colors.grey)
        canvas.drawString(2 * cm, 1.2 * cm, f"{doc_id} - version {version} - document fictif MaintiDoc")
        canvas.drawRightString(A4[0] - 2 * cm, 1.2 * cm, f"page {doc.page}")
        canvas.restoreState()

    target = str(path) if isinstance(path, Path) else path  # chemin ou flux mémoire
    SimpleDocTemplate(target, pagesize=A4, leftMargin=2 * cm, rightMargin=2 * cm,
                      topMargin=2 * cm, bottomMargin=2 * cm, title=doc_id).build(
        story, onFirstPage=footer, onLaterPages=footer)


def rl_image(pil_path, width_cm):
    w, h = Image.open(pil_path).size
    return RLImage(str(pil_path), width=width_cm * cm, height=width_cm * cm * h / w)


def seuil_valeur(famille, modele):
    s = FAMILLES[famille]["seuil"]
    return f"{FAMILLES[famille]['modeles'][modele][s['key']]} {s['unit']}"


# --------------------------------------------------------------------------- images d'équipements

def draw_panel(famille, modele, lit=None, code="", show_label=True):
    """Face avant : 4 voyants L1..L4, afficheur, boutons. Disposition identique pour les deux familles."""
    W, H = 900, 520
    img = Image.new("RGB", (W, H), (70, 74, 80))
    d = ImageDraw.Draw(img)
    d.rounded_rectangle((30, 30, W - 30, H - 30), 25, fill=(185, 190, 196), outline=(40, 40, 40), width=4)
    if show_label:
        d.text((70, 55), f"{FAMILLES[famille]['marque']}  {modele}", fill=(20, 20, 20), font=font(38))
    d.rectangle((560, 55, 830, 135), fill=(20, 35, 20), outline=(0, 0, 0), width=3)
    d.text((585, 68), code or "----", fill=(120, 255, 120) if code else (40, 80, 40), font=font(52))
    for i, (nom, couleur, _) in enumerate(FAMILLES[famille]["voyants"]):
        cx, cy, r = 140 + i * 190, 270, 45
        base = COULEURS[couleur]
        if nom == lit:
            for g in range(30, 0, -6):  # halo
                d.ellipse((cx - r - g, cy - r - g, cx + r + g, cy + r + g),
                          fill=tuple(min(255, int(c * 0.6 + 100)) for c in base))
            fill = base
        else:
            fill = tuple(int(c * 0.3) + 40 for c in base)
        d.ellipse((cx - r, cy - r, cx + r, cy + r), fill=fill, outline=(30, 30, 30), width=4)
        d.text((cx - 20, cy + 65), nom, fill=(20, 20, 20), font=font(30))
    # boutons : POWER (vert), ID (bleu), RESET (petit, gris)
    d.ellipse((120, 400, 200, 480), fill=(30, 150, 50), outline=(0, 0, 0), width=3)
    d.text((215, 425), "POWER", fill=(20, 20, 20), font=font(26))
    d.ellipse((420, 405, 490, 475), fill=(60, 110, 210), outline=(0, 0, 0), width=3)
    d.text((505, 425), "ID", fill=(20, 20, 20), font=font(26))
    d.ellipse((670, 418, 714, 462), fill=(120, 120, 125), outline=(0, 0, 0), width=3)
    d.text((728, 425), "RESET", fill=(20, 20, 20), font=font(26))
    return img


def draw_plaque(famille, modele, serie):
    fam = FAMILLES[famille]
    spec = fam["modeles"][modele]
    img = Image.new("RGB", (700, 400), (200, 200, 205))
    d = ImageDraw.Draw(img)
    d.rectangle((10, 10, 690, 390), outline=(60, 60, 60), width=5)
    d.text((40, 30), fam["marque"], fill=(10, 10, 10), font=font(40))
    lignes = [f"Modèle : {modele}", f"N° série : {serie}"]
    lignes += [f"{LIBELLES_PLAQUE[k]} : {spec[k]}" for k in fam["plaque"]]
    lignes += ["Année : 2024    Tension : 230 V / 50 Hz"]
    for i, l in enumerate(lignes):
        d.text((40, 100 + i * 52), l, fill=(20, 20, 20), font=font(30))
    return img


def degrade(img, blur=0.0, dark=1.0, noise=0.0, angle=0.0, glare=False):
    if glare:  # reflet qui masque le nom du modèle
        ImageDraw.Draw(img).ellipse((30, 20, 560, 140), fill=(250, 250, 250))
    if angle:
        img = img.rotate(angle, expand=True, fillcolor=(55, 55, 60))
    if blur:
        img = img.filter(ImageFilter.GaussianBlur(blur))
    if dark != 1.0:
        img = ImageEnhance.Brightness(img).enhance(dark)
    if noise:
        arr = np.asarray(img).astype(np.int16)
        img = Image.fromarray(np.clip(arr + np.random.normal(0, noise, arr.shape), 0, 255).astype(np.uint8))
    return img


# --------------------------------------------------------------------------- schémas du manuel pédagogique

def _box(d, xy, label, fill=(230, 236, 246), size=24):
    d.rounded_rectangle(xy, 12, fill=fill, outline=(40, 60, 100), width=3)
    x0, y0, x1, y1 = xy
    for k, line in enumerate(label.split("\n")):
        d.text(((x0 + x1) / 2, (y0 + y1) / 2 + (k - (label.count("\n")) / 2) * (size + 4)), line,
               fill=(20, 20, 30), font=font(size), anchor="mm")


def _canvas(title):
    img = Image.new("RGB", (1000, 560), "white")
    d = ImageDraw.Draw(img)
    d.text((500, 30), title, fill=(30, 50, 110), font=font(30), anchor="mm")
    return img, d


def schema_architecture():
    img, d = _canvas("Architecture d'un SAN")
    for i in range(3):
        _box(d, (60 + i * 310, 80, 300 + i * 310, 170), f"Serveur {i + 1}\nHBA")
    _box(d, (200, 260, 440, 340), "Switch FC\nfabric A", (255, 240, 220))
    _box(d, (560, 260, 800, 340), "Switch FC\nfabric B", (255, 240, 220))
    _box(d, (300, 430, 700, 530), "Baie StorNova\ncontrôleurs A et B", (225, 245, 225))
    for i in range(3):
        x = 180 + i * 310
        d.line((x, 170, 320, 260), fill=(40, 90, 200), width=3)
        d.line((x, 170, 680, 260), fill=(230, 130, 20), width=3)
    d.line((320, 340, 430, 430), fill=(40, 90, 200), width=3)
    d.line((680, 340, 570, 430), fill=(230, 130, 20), width=3)
    return img


def schema_multipath():
    img, d = _canvas("Double fabric et multipath")
    _box(d, (380, 70, 620, 160), "Serveur\nHBA 1   HBA 2")
    _box(d, (100, 250, 340, 330), "Fabric A", (220, 232, 255))
    _box(d, (660, 250, 900, 330), "Fabric B", (255, 236, 215))
    _box(d, (100, 430, 340, 520), "Contrôleur A", (225, 245, 225))
    _box(d, (660, 430, 900, 520), "Contrôleur B", (225, 245, 225))
    d.line((440, 160, 220, 250), fill=(40, 90, 200), width=5)
    d.line((560, 160, 780, 250), fill=(230, 130, 20), width=5)
    d.line((220, 330, 220, 430), fill=(40, 90, 200), width=5)
    d.line((780, 330, 780, 430), fill=(230, 130, 20), width=5)
    d.text((500, 380), "si une fabric tombe,\nle trafic passe par l'autre", fill=(90, 90, 90), font=font(22),
           anchor="mm", align="center")
    return img


def schema_raid():
    img, d = _canvas("Groupe RAID 6 : 6 disques de données + 2 de parité")
    for i in range(8):
        x = 60 + i * 115
        parite = i >= 6
        fill = (255, 225, 200) if parite else (220, 232, 255)
        d.rounded_rectangle((x, 140, x + 95, 400), 10, fill=fill, outline=(40, 60, 100), width=3)
        d.text((x + 47, 270), ("P" if i == 6 else "Q") if parite else f"D{i + 1}", fill=(20, 20, 30),
               font=font(34), anchor="mm")
    d.text((500, 460), "Jusqu'à 2 disques perdus sans perte de données", fill=(30, 30, 30), font=font(26),
           anchor="mm")
    return img


def schema_zoning():
    img, d = _canvas("Zoning : quels ports communiquent")
    _box(d, (380, 220, 620, 340), "Switch FC", (255, 240, 220))
    zones = [((40, 90, 320, 520), (40, 90, 200), "Zone 1", ["Serveur 1 (WWN ..:A1)", "Baie port 0"]),
             ((680, 90, 960, 520), (30, 150, 70), "Zone 2", ["Serveur 2 (WWN ..:B7)", "Baie port 1"])]
    for (x0, y0, x1, y1), col, name, members in zones:
        for k in range(0, x1 - x0, 24):  # bordure en pointillés
            d.line((x0 + k, y0, x0 + k + 12, y0), fill=col, width=4)
            d.line((x0 + k, y1, x0 + k + 12, y1), fill=col, width=4)
        d.text(((x0 + x1) / 2, y0 + 30), name, fill=col, font=font(28), anchor="mm")
        for j, m in enumerate(members):
            _box(d, (x0 + 20, y0 + 90 + j * 170, x1 - 20, y0 + 200 + j * 170), m, size=20)
    return img


def photo_couverture():
    """Photo de couverture : baie informatique avec switchs et baie de stockage."""
    img = Image.new("RGB", (900, 600), (30, 32, 36))
    d = ImageDraw.Draw(img)
    d.rectangle((250, 30, 650, 580), fill=(15, 15, 18), outline=(80, 80, 85), width=6)
    y = 60
    for kind in ("switch", "switch", "vide", "baie", "baie", "vide", "baie"):
        h = 40 if kind == "switch" else (90 if kind == "baie" else 30)
        if kind != "vide":
            d.rectangle((270, y, 630, y + h), fill=(55, 58, 64), outline=(100, 100, 110), width=2)
            if kind == "switch":
                for p in range(16):
                    x = 290 + p * 20
                    d.rectangle((x, y + 12, x + 12, y + 28), fill=(20, 20, 20))
                    d.ellipse((x + 3, y + 3, x + 9, y + 9), fill=(40, 220, 80) if p % 5 else (255, 160, 30))
            else:
                for b in range(12):
                    x = 285 + b * 28
                    d.rectangle((x, y + 10, x + 22, y + h - 10), fill=(85, 88, 95), outline=(30, 30, 30))
                    d.ellipse((x + 8, y + h - 22, x + 14, y + h - 16), fill=(40, 220, 80))
        y += h + 12
    img = img.filter(ImageFilter.GaussianBlur(0.8))
    return degrade(img, noise=8)


SCHEMAS = {"architecture": (schema_architecture, "Architecture d'un SAN : serveurs, switchs et baie"),
           "multipath": (schema_multipath, "Double fabric (A et B) et multipath"),
           "raid": (schema_raid, "Groupe RAID 6"),
           "zoning": (schema_zoning, "Exemple de zoning sur un switch Fibre Channel"),
           "panneau": (lambda: draw_panel("baie", "SN-400"), "Face avant d'une baie StorNova SN-400")}


# --------------------------------------------------------------------------- documents

def manuel_pedagogique():
    m = MANUEL_SAN
    doc_id = m["id"]
    cover = FIGURES / f"{doc_id}_couverture.png"
    photo_couverture().save(cover)
    s = [Paragraph(m["titre"], H1), Paragraph("Document fictif à usage pédagogique - projet MaintiDoc", CAPTION),
         Spacer(1, 20), rl_image(cover, 13), Paragraph("Figure 0 - Baie informatique d'un SAN", CAPTION),
         PageBreak()]
    for n, (titre, paragraphes, schema) in enumerate(m["sections"], 1):
        draw, legende = SCHEMAS[schema]
        fig = FIGURES / f"{doc_id}_{schema}.png"
        draw().save(fig)
        s += [Paragraph(titre, H2)] + [Paragraph(p, BODY) for p in paragraphes]
        s += [Spacer(1, 8), rl_image(fig, 14), Paragraph(f"Figure {n} - {legende}", CAPTION), Spacer(1, 8)]
        if schema == "raid":
            s += [table(m["raid"], [3 * cm, 5 * cm, 9 * cm])]
        s.append(PageBreak())
    s += [Paragraph("6. Bonnes pratiques de maintenance", H2),
          Paragraph("Avant toute intervention sur une baie, appliquer la procédure SEC-B01 ; sur un switch, "
                    "la procédure SEC-S01. Vérifier la redondance, sauvegarder la configuration, intervenir sur "
                    "un seul composant ou une seule fabric à la fois, et tracer l'intervention.", BODY),
          Spacer(1, 10), Paragraph("Glossaire", H2), table(m["glossaire"], [4 * cm, 13 * cm])]
    path = CORPUS / f"{doc_id}.pdf"
    build_pdf(path, doc_id, "v1", s)
    register(doc_id, m["titre"], "san", "manuel_pedagogique", path, groupes=TOUS_LES_GROUPES)


def guide_mcp():
    """Guide d'utilisation de MaintiDoc (outils MCP, validation humaine, tickets), accessible à tous."""
    g = GUIDE_MCP
    s = [Paragraph(g["titre"], H1), Paragraph("Document fictif - projet MaintiDoc", CAPTION), Spacer(1, 10)]
    for titre, avant, tableau, apres in g["sections"]:
        s += [Paragraph(titre, H2)] + [Paragraph(p, BODY) for p in avant]
        if tableau:
            s += [Spacer(1, 4), table(tableau, [4.5 * cm, 12.5 * cm]), Spacer(1, 4)]
        s += [Paragraph(p, BODY) for p in apres] + [Spacer(1, 6)]
    path = CORPUS / f"{g['id']}.pdf"
    build_pdf(path, g["id"], "v1", s)
    register(g["id"], g["titre"], "san", "guide", path, groupes=TOUS_LES_GROUPES)


def manuel(famille, modele):
    fam = FAMILLES[famille]
    spec = fam["modeles"][modele]
    seuil = fam["seuil"]
    doc_id = f"manuel_{famille}_{modele.lower()}"
    fig_panel = FIGURES / f"{doc_id}_panneau.png"
    draw_panel(famille, modele).save(fig_panel)
    fig_plaque = FIGURES / f"{doc_id}_plaque.png"
    draw_plaque(famille, modele, "XX-0000-0000").save(fig_plaque)

    s = [Paragraph(f"Manuel d'utilisation - {fam['marque']} {modele}", H1),
         Paragraph(f"{fam['libelle']} - document fictif à usage de démonstration", CAPTION), Spacer(1, 12),
         Paragraph("1. Présentation", H2),
         Paragraph(fam["presentation"].format(modele=modele, **spec), BODY),
         Paragraph("2. Consignes de sécurité", H2),
         Paragraph(f"Toute intervention doit respecter la procédure {fam['sec_ref']}. Seul un personnel habilité "
                   "peut intervenir sur l'équipement. Ce manuel ne remplace pas les consignes du site.", BODY),
         PageBreak(),
         Paragraph("3. Panneau de contrôle", H2),
         Paragraph("La face avant comporte quatre voyants (L1 à L4), un afficheur de code et les boutons "
                   "POWER, ID et RESET.", BODY), Spacer(1, 8),
         rl_image(fig_panel, 14),
         Paragraph(f"Figure 3.1 - Face avant du {modele}", CAPTION), Spacer(1, 10),
         table([["Voyant", "Couleur", "Signification"]] + [list(v) for v in fam["voyants"]],
               [2 * cm, 2.5 * cm, 12.5 * cm]),
         Spacer(1, 8),
         Paragraph(f"Le seuil de l'alarme {seuil['code']} ({seuil['label']}) est fixé à "
                   f"{seuil_valeur(famille, modele)}.", BODY),
         PageBreak(),
         Paragraph("4. Mise sous tension et arrêt", H2),
         Paragraph(fam["demarrage"], BODY), Paragraph(fam["arret"], BODY),
         Paragraph("5. Plaque signalétique", H2),
         Paragraph("La plaque signalétique est fixée à l'arrière de l'équipement. Elle indique le modèle et le "
                   "numéro de série, à communiquer pour toute demande d'assistance.", BODY), Spacer(1, 8),
         rl_image(fig_plaque, 10),
         Paragraph(f"Figure 5.1 - Exemple de plaque signalétique {modele}", CAPTION),
         Paragraph("6. Codes d'alarme", H2),
         Paragraph(f"La liste complète des codes figure dans le document alarmes_{famille}_{modele.lower()}.", BODY)]
    path = CORPUS / f"{doc_id}.pdf"
    build_pdf(path, doc_id, "v1", s)
    register(doc_id, f"Manuel d'utilisation {modele}", famille, "manuel", path, modele=modele)


def alarmes(famille, modele):
    fam = FAMILLES[famille]
    doc_id = f"alarmes_{famille}_{modele.lower()}"
    rows = [["Code", "Libellé", "Voyant", "Cause probable", "Action"]]
    for code, lib, voy, cause, action in fam["alarmes"]:
        if code == fam["seuil"]["code"]:
            lib = f"{lib} (seuil {seuil_valeur(famille, modele)})"
        rows.append([code, lib, voy, cause, action])
    s = [Paragraph(f"Tableau des alarmes - {fam['marque']} {modele}", H1), Spacer(1, 8),
         Paragraph("En cas de doute, ou si une alarme bloquante se répète, ne rien redémarrer et contacter le "
                   "responsable de maintenance.", BODY), Spacer(1, 8),
         table(rows, [1.4 * cm, 3.4 * cm, 2 * cm, 4.6 * cm, 5.6 * cm])]
    path = CORPUS / f"{doc_id}.pdf"
    build_pdf(path, doc_id, "v1", s)
    register(doc_id, f"Tableau des alarmes {modele}", famille, "alarmes", path, modele=modele)


def fiche(famille, modele):
    fam = FAMILLES[famille]
    seuil_key = fam["seuil"]["key"]
    doc_id = f"fiche_{famille}_{modele.lower()}"
    rows = [["Caractéristique", "Valeur"]]
    for k, v in fam["modeles"][modele].items():
        rows.append([k.replace("_", " ").capitalize(), f"{v} {fam['seuil']['unit']}" if k == seuil_key else v])
    s = [Paragraph(f"Fiche technique - {fam['marque']} {modele}", H1), Spacer(1, 10),
         table(rows, [7 * cm, 7 * cm]), Spacer(1, 10),
         Paragraph("Valeurs nominales données à titre indicatif pour le prototype MaintiDoc.", SMALL)]
    path = CORPUS / f"{doc_id}.pdf"
    build_pdf(path, doc_id, "v1", s)
    register(doc_id, f"Fiche technique {modele}", famille, "fiche_technique", path, modele=modele)


def procedure(famille, pid, titre, freq, outillage, etapes):
    doc_id = f"procedure_{famille}_{pid.lower()}"
    s = [Paragraph(f"{pid} - {titre}", H1), Spacer(1, 8),
         table([["Fréquence", freq], ["Outillage", outillage],
                ["Modèles", ", ".join(FAMILLES[famille]["modeles"])]], [4 * cm, 13 * cm], header=False),
         Spacer(1, 10), Paragraph("Étapes", H2)]
    s += [Paragraph(f"{i}. {e}", BODY) for i, e in enumerate(etapes, 1)]
    path = CORPUS / f"{doc_id}.pdf"
    build_pdf(path, doc_id, "v1", s)
    register(doc_id, f"{pid} {titre}", famille, "procedure", path)


def securite(famille, sid, titre, etapes, risques):
    doc_id = f"securite_{famille}_{sid.lower()}"
    s = [Paragraph(f"{sid} - {titre}", H1), Spacer(1, 8), Paragraph("Risques spécifiques", H2)]
    s += [Paragraph(f"- {r}", BODY) for r in risques]
    s += [Paragraph("Mesures", H2)] + [Paragraph(f"{i}. {e}", BODY) for i, e in enumerate(etapes, 1)]
    path = CORPUS / f"{doc_id}.pdf"
    build_pdf(path, doc_id, "v1", s)
    # Les documents de sécurité sont accessibles à tous les groupes.
    register(doc_id, f"{sid} {titre}", famille, "securite", path, groupes=TOUS_LES_GROUPES)


def note(famille, n):
    s = [Paragraph(n["titre"], H1), Paragraph(f"Date d'effet : {n['date']}", CAPTION), Spacer(1, 10)]
    s += [Paragraph(p, BODY) for p in n["paragraphes"]]
    path = CORPUS / f"{n['id']}.pdf"
    build_pdf(path, n["id"], "v1", s)
    register(n["id"], n["titre"], famille, "note_service", path, restreint=n["restreint"],
             date_effet=n["date"], contredit=n["contredit"], test_injection=n.get("injection", False))


def scan(famille, doc_id, titre, lignes):
    tmp = io.BytesIO()
    s = [Paragraph(titre, H1), Paragraph("Document d'archive (2008) - recopié pour numérisation", CAPTION),
         Spacer(1, 12)] + [Paragraph(f"{i}. {l}", BODY) for i, l in enumerate(lignes, 1)]
    build_pdf(tmp, doc_id, "v1", s)
    pages = []
    with pymupdf.open(stream=tmp.getvalue(), filetype="pdf") as pdf:
        for page in pdf:
            pix = page.get_pixmap(dpi=110)
            img = Image.frombytes("RGB", (pix.width, pix.height), pix.samples).convert("L").convert("RGB")
            img = Image.blend(img, Image.new("RGB", img.size, (235, 225, 200)), 0.25)  # papier jauni
            pages.append(degrade(img, blur=0.6, noise=14, angle=random.uniform(-2.0, 2.0)))
    path = CORPUS / f"{doc_id}.pdf"
    pages[0].save(path, "PDF", resolution=110, save_all=True, append_images=pages[1:])
    register(doc_id, titre, famille, "scan", path, scanne=True)


# --------------------------------------------------------------------------- photos

def photos():
    meta = []
    n = 0

    def save(img, **truth):
        nonlocal n
        n += 1
        name = f"photo_{n:02d}.jpg"
        img.convert("RGB").save(PHOTOS / name, quality=80)
        meta.append({"fichier": name, **truth})

    for famille, fam in FAMILLES.items():
        codes_bloquants = [a[0] for a in fam["alarmes"] if a[2].startswith("L3")]
        for modele in fam["modeles"]:
            save(degrade(draw_panel(famille, modele, lit="L2"), blur=0.8, noise=6, angle=random.uniform(-6, 6)),
                 famille=famille, modele=modele, voyant="L2", code=None, lisible=True, ambigu=False)
            code = random.choice(codes_bloquants)
            save(degrade(draw_panel(famille, modele, lit="L3", code=code), blur=1.0, noise=8,
                         dark=0.85, angle=random.uniform(-6, 6)),
                 famille=famille, modele=modele, voyant="L3", code=code, lisible=True, ambigu=False)
            save(degrade(draw_panel(famille, modele, lit="L1"), blur=1.2, noise=6, angle=random.uniform(-8, 8)),
                 famille=famille, modele=modele, voyant="L1", code=None, lisible=True, ambigu=False)
            # Ambiguë : reflet sur le nom du modèle, voyant orange -> impossible de savoir baie ou switch.
            save(degrade(draw_panel(famille, modele, lit="L2"), blur=2.5, noise=10, dark=0.6, glare=True,
                         angle=random.uniform(-10, 10)),
                 famille=famille, modele=modele, voyant="L2", code=None, lisible=False, ambigu=True,
                 attendu="needs_clarification")
        for i, modele in enumerate(fam["modeles"]):
            serie = f"{fam['code_prefix']}{modele.replace('-', '')}-2024-{1000 + i}"
            illisible = i == 2
            img = degrade(draw_plaque(famille, modele, serie), blur=6 if illisible else 0.7,
                          noise=20 if illisible else 5, dark=0.5 if illisible else 1.0, angle=random.uniform(-5, 5))
            save(img, famille=famille, modele=modele, type="plaque_signaletique", lisible=not illisible,
                 ambigu=illisible, attendu="needs_clarification" if illisible else "answered")
    (PHOTOS / "photos.json").write_text(json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8")
    return len(meta)


# --------------------------------------------------------------------------- base MCP

def base_mcp():
    db_path = MCP / "maintenance.db"
    db_path.unlink(missing_ok=True)
    con = sqlite3.connect(db_path)
    con.executescript("""
        CREATE TABLE equipements (equipment_id TEXT PRIMARY KEY, famille TEXT, modele TEXT,
                                  site TEXT, mise_en_service TEXT, groupe TEXT);
        CREATE TABLE interventions (id INTEGER PRIMARY KEY, equipment_id TEXT, date TEXT, type TEXT,
                                    code_alarme TEXT, description TEXT, technicien TEXT, duree_h REAL);
        CREATE TABLE tickets (ticket_id TEXT PRIMARY KEY, equipment_id TEXT, titre TEXT, description TEXT,
                              statut TEXT, cree_par TEXT, cree_le TEXT, idempotency_key TEXT UNIQUE);
    """)
    equipements = []
    for famille, fam in FAMILLES.items():
        modeles = list(fam["modeles"])
        k = 0
        for modele in modeles:
            for _ in range(2 if modele == modeles[1] else 1):   # le modèle intermédiaire est le plus déployé
                k += 1
                eid = f"{fam['id_prefix']}-{modele.replace('-', '')}-{k:03d}"
                equipements.append((eid, famille, modele, random.choice(["Salle serveurs A", "Salle serveurs B"]),
                                    f"202{random.randint(1, 4)}-0{random.randint(1, 9)}-15", fam["groupe"]))
    con.executemany("INSERT INTO equipements VALUES (?,?,?,?,?,?)", equipements)

    rows, start = [], date(2025, 6, 1)
    for eid, famille, *_ in equipements:
        fam = FAMILLES[famille]
        for _ in range(random.randint(4, 7)):
            d = start + timedelta(days=random.randint(0, 500))
            if random.random() < 0.5:
                code, lib, *_rest, action = random.choice(fam["alarmes"])
                rows.append((eid, d.isoformat(), "corrective", code, f"Alarme {code} ({lib}). {action}",
                             random.choice(fam["techs"]), round(random.uniform(0.5, 4), 1)))
            else:
                pid, titre, *_r = random.choice(fam["procedures"])
                rows.append((eid, d.isoformat(), "preventive", None, f"{pid} {titre} réalisée.",
                             random.choice(fam["techs"]), round(random.uniform(0.5, 3), 1)))
    rows.sort(key=lambda r: r[1])
    con.executemany("INSERT INTO interventions (equipment_id,date,type,code_alarme,description,technicien,"
                    "duree_h) VALUES (?,?,?,?,?,?,?)", rows)
    con.executemany("INSERT INTO tickets VALUES (?,?,?,?,?,?,?,?)", [
        ("TCK-0001", "BAIE-SN400-002", "Disque défaillant", "Alarme E01 sur le tiroir 7.",
         "ouvert", "tech_s01", "2026-10-20", "seed-1"),
        ("TCK-0002", "SW-FL48-002", "Erreurs CRC port 12", "Alarme F01 récurrente sur le port 12.",
         "ouvert", "tech_r02", "2026-10-28", "seed-2"),
        ("TCK-0003", "BAIE-SN200-001", "Remplacement batterie de cache", "Alarme E05, batterie remplacée.",
         "ferme", "tech_s02", "2026-08-02", "seed-3"),
    ])
    con.commit()
    con.close()
    return len(equipements), len(rows)


def utilisateurs():
    users = [
        {"user_id": "tech_s01", "nom": "Technicien Stockage 1", "groupes": ["tech_stockage"]},
        {"user_id": "tech_s02", "nom": "Technicien Stockage 2", "groupes": ["tech_stockage"]},
        {"user_id": "tech_r01", "nom": "Technicien Réseau SAN 1", "groupes": ["tech_reseau"]},
        {"user_id": "tech_r02", "nom": "Technicien Réseau SAN 2", "groupes": ["tech_reseau"]},
        {"user_id": "resp_01", "nom": "Responsable Maintenance", "groupes": ["responsable"]},
    ]
    # Comptes de démonstration : mot de passe « <identifiant>-demo », stocké uniquement sous forme hachée.
    # Sel dérivé de l'identifiant pour que le fichier reste identique d'une génération à l'autre.
    for u in users:
        u["password_hash"] = hash_password(f"{u['user_id']}-demo", salt=hashlib.sha256(u["user_id"].encode()).digest()[:16])
    (ROOT / "users.json").write_text(json.dumps(users, ensure_ascii=False, indent=2), encoding="utf-8")


# --------------------------------------------------------------------------- main

def main():
    for p in (CORPUS, FIGURES, PHOTOS, MCP):
        p.mkdir(parents=True, exist_ok=True)
    if "--mcp-only" in sys.argv:   # la base MCP n'est pas versionnée : la recréer sans toucher aux PDF
        nb_eq, nb_int = base_mcp()
        print(f"Base MCP : {nb_eq} équipements, {nb_int} interventions, 3 tickets -> {MCP / 'maintenance.db'}")
        return
    # On repart d'un corpus vide : les fichiers d'un ancien domaine ne doivent pas subsister.
    for p in (*CORPUS.glob("*.pdf"), *FIGURES.glob("*.png"), *PHOTOS.glob("*.jpg")):
        p.unlink()

    manuel_pedagogique()
    guide_mcp()
    for famille, fam in FAMILLES.items():
        for modele in fam["modeles"]:
            manuel(famille, modele)
            alarmes(famille, modele)
            fiche(famille, modele)
        for pid, titre, freq, outillage, etapes in fam["procedures"]:
            procedure(famille, pid, titre, freq, outillage, etapes)
        for sid, titre, etapes, risques in SECURITE[famille]:
            securite(famille, sid, titre, etapes, risques)
        for n in NOTES[famille]:
            note(famille, n)
        for doc_id, titre, lignes in SCANS[famille]:
            scan(famille, doc_id, titre, lignes)

    (CORPUS / "metadata.json").write_text(json.dumps(documents, ensure_ascii=False, indent=2), encoding="utf-8")
    nb_photos = photos()
    nb_eq, nb_int = base_mcp()
    utilisateurs()

    print(f"Documents PDF : {len(documents)}  -> {CORPUS}")
    print(f"Photos        : {nb_photos}  -> {PHOTOS}")
    print(f"Base MCP      : {nb_eq} équipements, {nb_int} interventions, 3 tickets -> {MCP / 'maintenance.db'}")
    print(f"Utilisateurs  : {ROOT / 'users.json'}")


if __name__ == "__main__":
    main()
