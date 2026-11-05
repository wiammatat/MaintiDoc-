"""Génère le corpus synthétique MaintiDoc dans data/.

Usage : python scripts/generate_corpus.py
Toutes les données sont fictives. Le générateur est déterministe (graine fixe).
"""

import io
import json
import random
import sqlite3
import sys
from datetime import date, timedelta
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFilter, ImageFont, ImageEnhance
from reportlab.lib import colors
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.units import cm
from reportlab.platypus import (Image as RLImage, PageBreak, Paragraph, SimpleDocTemplate,
                                Spacer, Table, TableStyle)

try:
    import pymupdf
except ImportError:
    import fitz as pymupdf

sys.path.insert(0, str(Path(__file__).parent))
from corpus_content import FAMILLES, NOTES, SCANS, SECURITE  # noqa: E402

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


# --------------------------------------------------------------------------- images

def draw_panel(famille, modele, lit=None, code="", show_label=True):
    """Panneau de commande : 4 voyants L1..L4, afficheur, boutons. Disposition identique pour les deux familles."""
    W, H = 900, 520
    img = Image.new("RGB", (W, H), (70, 74, 80))
    d = ImageDraw.Draw(img)
    d.rounded_rectangle((30, 30, W - 30, H - 30), 25, fill=(185, 190, 196), outline=(40, 40, 40), width=4)
    if show_label:
        d.text((70, 55), f"{FAMILLES[famille]['marque']}  {modele}", fill=(20, 20, 20), font=font(38))
    # afficheur
    d.rectangle((560, 55, 830, 135), fill=(20, 35, 20), outline=(0, 0, 0), width=3)
    d.text((585, 68), code or "----", fill=(120, 255, 120) if code else (40, 80, 40), font=font(52))
    # voyants
    for i, (nom, couleur, _) in enumerate(FAMILLES[famille]["voyants"]):
        cx, cy, r = 140 + i * 190, 270, 45
        base = COULEURS[couleur]
        if nom == lit:
            for g in range(30, 0, -6):  # halo
                halo = tuple(min(255, int(c * 0.6 + 100)) for c in base)
                d.ellipse((cx - r - g, cy - r - g, cx + r + g, cy + r + g), fill=halo)
            fill = base
        else:
            fill = tuple(int(c * 0.3) + 40 for c in base)
        d.ellipse((cx - r, cy - r, cx + r, cy + r), fill=fill, outline=(30, 30, 30), width=4)
        d.text((cx - 20, cy + 65), nom, fill=(20, 20, 20), font=font(30))
    # boutons
    d.ellipse((120, 400, 200, 480), fill=(30, 150, 50), outline=(0, 0, 0), width=3)
    d.text((215, 425), "START", fill=(20, 20, 20), font=font(26))
    d.ellipse((380, 400, 460, 480), fill=(200, 30, 30), outline=(0, 0, 0), width=3)
    d.text((475, 425), "STOP", fill=(20, 20, 20), font=font(26))
    d.ellipse((680, 385, 790, 495), fill=(220, 190, 20), outline=(0, 0, 0), width=3)
    d.ellipse((700, 405, 770, 475), fill=(200, 20, 20))
    return img


def draw_plaque(famille, modele, serie):
    spec = FAMILLES[famille]["modeles"][modele]
    img = Image.new("RGB", (700, 400), (200, 200, 205))
    d = ImageDraw.Draw(img)
    d.rectangle((10, 10, 690, 390), outline=(60, 60, 60), width=5)
    d.text((40, 30), FAMILLES[famille]["marque"], fill=(10, 10, 10), font=font(40))
    lignes = [f"Modèle : {modele}", f"N° série : {serie}", f"Puissance : {spec['puissance']}",
              f"Débit : {spec['debit']}", "Année : 2024    Tension : 400 V / 50 Hz"]
    for i, l in enumerate(lignes):
        d.text((40, 100 + i * 52), l, fill=(20, 20, 20), font=font(30))
    return img


def degrade(img, blur=0.0, dark=1.0, noise=0.0, angle=0.0, glare=False):
    if glare:  # reflet qui masque le nom du modèle
        d = ImageDraw.Draw(img)
        d.ellipse((30, 20, 560, 140), fill=(250, 250, 250))
    if angle:
        img = img.rotate(angle, expand=True, fillcolor=(55, 55, 60))
    if blur:
        img = img.filter(ImageFilter.GaussianBlur(blur))
    if dark != 1.0:
        img = ImageEnhance.Brightness(img).enhance(dark)
    if noise:
        arr = np.asarray(img).astype(np.int16)
        arr = np.clip(arr + np.random.normal(0, noise, arr.shape), 0, 255).astype(np.uint8)
        img = Image.fromarray(arr)
    return img


# --------------------------------------------------------------------------- documents

def manuel(famille, modele):
    fam = FAMILLES[famille]
    spec = fam["modeles"][modele]
    slug = modele.lower()
    doc_id = f"manuel_{famille}_{slug}"
    fig_panel = FIGURES / f"{doc_id}_panneau.png"
    draw_panel(famille, modele).save(fig_panel)
    fig_plaque = FIGURES / f"{doc_id}_plaque.png"
    draw_plaque(famille, modele, "XX-0000-0000").save(fig_plaque)

    seuil = (f"Le seuil de l'alarme E01 (surchauffe moteur) est fixé à {spec['seuil_temp']} °C."
             if famille == "pompe" else
             f"Le seuil de l'alarme A01 (pression haute) est fixé à {spec['seuil_pression']}.")
    s = [Paragraph(f"Manuel d'utilisation - {fam['marque']} {modele}", H1),
         Paragraph(f"{fam['libelle']} - document fictif à usage de démonstration", CAPTION), Spacer(1, 12),
         Paragraph("1. Présentation", H2),
         Paragraph(f"Le {modele} appartient à la gamme {fam['libelle'].lower()} {fam['marque']}. "
                   f"Puissance nominale : {spec['puissance']}, débit nominal : {spec['debit']}.", BODY),
         Paragraph("2. Consignes de sécurité", H2),
         Paragraph("Toute intervention nécessite une consignation préalable selon la procédure "
                   f"{'SEC-P01' if famille == 'pompe' else 'SEC-C01'}. Seul un personnel habilité peut "
                   "intervenir sur l'équipement. Ce manuel ne remplace pas les consignes du site.", BODY),
         PageBreak(),
         Paragraph("3. Panneau de commande", H2),
         Paragraph("Le panneau comporte quatre voyants (L1 à L4), un afficheur de code, les boutons START "
                   "et STOP et un bouton d'arrêt d'urgence.", BODY), Spacer(1, 8),
         rl_image(fig_panel, 14),
         Paragraph(f"Figure 3.1 - Panneau de commande du {modele}", CAPTION), Spacer(1, 10),
         table([["Voyant", "Couleur", "Signification"]] + [list(v) for v in fam["voyants"]],
               [2 * cm, 2.5 * cm, 12.5 * cm]),
         Spacer(1, 8), Paragraph(seuil, BODY),
         PageBreak(),
         Paragraph("4. Démarrage et arrêt", H2),
         Paragraph("Démarrage : vérifier qu'aucun code n'est affiché, puis appuyer sur START. Le voyant L1 "
                   "vert doit s'allumer en moins de 10 secondes.", BODY),
         Paragraph("Arrêt : appuyer sur STOP. Le bouton d'arrêt d'urgence ne doit être utilisé qu'en cas de "
                   "danger : il déclenche l'alarme " + ("E10." if famille == "pompe" else "A10."), BODY),
         Paragraph("5. Plaque signalétique", H2),
         Paragraph("La plaque signalétique est fixée sur le côté du carter. Elle indique le modèle et le "
                   "numéro de série, à communiquer pour toute demande d'assistance.", BODY), Spacer(1, 8),
         rl_image(fig_plaque, 10),
         Paragraph(f"Figure 5.1 - Exemple de plaque signalétique {modele}", CAPTION),
         Paragraph("6. Codes d'alarme", H2),
         Paragraph(f"La liste complète des codes figure dans le document alarmes_{famille}_{slug}.", BODY)]
    path = CORPUS / f"{doc_id}.pdf"
    build_pdf(path, doc_id, "v1", s)
    register(doc_id, f"Manuel d'utilisation {modele}", famille, "manuel", path, modele=modele)


def alarmes(famille, modele):
    fam = FAMILLES[famille]
    spec = fam["modeles"][modele]
    doc_id = f"alarmes_{famille}_{modele.lower()}"
    rows = [["Code", "Libellé", "Voyant", "Cause probable", "Action"]]
    for code, lib, voy, cause, action in fam["alarmes"]:
        if code == "E01":
            lib = f"{lib} (seuil {spec['seuil_temp']} °C)"
        if code == "A01":
            lib = f"{lib} (seuil {spec['seuil_pression']})"
        rows.append([code, lib, voy, cause, action])
    s = [Paragraph(f"Tableau des alarmes - {fam['marque']} {modele}", H1), Spacer(1, 8),
         Paragraph("En cas de doute, ou si une alarme bloquante se répète plus de 3 fois, ne pas réarmer et "
                   "contacter le responsable de maintenance.", BODY), Spacer(1, 8),
         table(rows, [1.4 * cm, 3.4 * cm, 2 * cm, 4.6 * cm, 5.6 * cm])]
    path = CORPUS / f"{doc_id}.pdf"
    build_pdf(path, doc_id, "v1", s)
    register(doc_id, f"Tableau des alarmes {modele}", famille, "alarmes", path, modele=modele)


def fiche(famille, modele):
    fam = FAMILLES[famille]
    spec = fam["modeles"][modele]
    doc_id = f"fiche_{famille}_{modele.lower()}"
    rows = [["Caractéristique", "Valeur"]] + [[k.replace("_", " ").capitalize(), v] for k, v in spec.items()]
    rows = [[r[0], f"{r[1]} °C" if r[0] == "Seuil temp" else r[1]] for r in rows]
    s = [Paragraph(f"Fiche technique - {fam['marque']} {modele}", H1), Spacer(1, 10),
         table(rows, [7 * cm, 7 * cm]), Spacer(1, 10),
         Paragraph("Valeurs nominales données à titre indicatif pour le prototype MaintiDoc.", SMALL)]
    path = CORPUS / f"{doc_id}.pdf"
    build_pdf(path, doc_id, "v1", s)
    register(doc_id, f"Fiche technique {modele}", famille, "fiche_technique", path, modele=modele)


def procedure(famille, pid, titre, freq, outillage, etapes, intro=None):
    doc_id = f"procedure_{famille}_{pid.lower()}"
    s = [Paragraph(f"{pid} - {titre}", H1), Spacer(1, 8),
         table([["Fréquence", freq], ["Outillage", outillage],
                ["Modèles", ", ".join(FAMILLES[famille]["modeles"])]], [4 * cm, 13 * cm], header=False),
         Spacer(1, 10), Paragraph("Étapes", H2)]
    if intro:
        s.append(Paragraph(intro, BODY))
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
    register(doc_id, f"{sid} {titre}", famille, "securite", path,
             groupes=["tech_pompes", "tech_compresseurs", "responsable"])


def note(famille, n):
    s = [Paragraph(n["titre"], H1), Paragraph(f"Date d'effet : {n['date']}", CAPTION), Spacer(1, 10)]
    s += [Paragraph(p, BODY) for p in n["paragraphes"]]
    path = CORPUS / f"{n['id']}.pdf"
    build_pdf(path, n["id"], "v1", s)
    register(n["id"], n["titre"], famille, "note_service", path, restreint=n["restreint"],
             date_effet=n["date"], contredit=n["contredit"], test_injection=n.get("injection", False))


def scan(famille, doc_id, titre, lignes):
    tmp = io.BytesIO()
    s = [Paragraph(titre, H1), Paragraph("Document d'archive (1998) - recopié pour numérisation", CAPTION),
         Spacer(1, 12)] + [Paragraph(f"{i}. {l}", BODY) for i, l in enumerate(lignes, 1)]
    build_pdf(tmp, doc_id, "v1", s)
    pages = []
    with pymupdf.open(stream=tmp.getvalue(), filetype="pdf") as pdf:
        for page in pdf:
            pix = page.get_pixmap(dpi=110)
            img = Image.frombytes("RGB", (pix.width, pix.height), pix.samples).convert("L").convert("RGB")
            img = ImageEnhance.Color(img).enhance(0)
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
        prefix = "E" if famille == "pompe" else "A"
        for modele in fam["modeles"]:
            save(degrade(draw_panel(famille, modele, lit="L2"), blur=0.8, noise=6, angle=random.uniform(-6, 6)),
                 famille=famille, modele=modele, voyant="L2", code=None, lisible=True, ambigu=False)
            code = f"{prefix}0{random.choice([3, 6, 7])}"
            save(degrade(draw_panel(famille, modele, lit="L3", code=code), blur=1.0, noise=8,
                         dark=0.85, angle=random.uniform(-6, 6)),
                 famille=famille, modele=modele, voyant="L3", code=code, lisible=True, ambigu=False)
            save(degrade(draw_panel(famille, modele, lit="L1"), blur=1.2, noise=6, angle=random.uniform(-8, 8)),
                 famille=famille, modele=modele, voyant="L1", code=None, lisible=True, ambigu=False)
            # Ambiguë : reflet sur le nom du modèle, voyant orange -> impossible de savoir pompe ou compresseur.
            save(degrade(draw_panel(famille, modele, lit="L2"), blur=2.5, noise=10, dark=0.6, glare=True,
                         angle=random.uniform(-10, 10)),
                 famille=famille, modele=modele, voyant="L2", code=None, lisible=False, ambigu=True,
                 attendu="needs_clarification")
        for i, modele in enumerate(fam["modeles"]):
            serie = f"{prefix}{modele[-3:].replace('-', '')}-2024-{1000 + i}"
            illisible = i == 2
            img = draw_plaque(famille, modele, serie)
            img = degrade(img, blur=6 if illisible else 0.7, noise=20 if illisible else 5,
                          dark=0.5 if illisible else 1.0, angle=random.uniform(-5, 5))
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
        prefix = "POMPE" if famille == "pompe" else "COMP"
        k = 0
        for modele in fam["modeles"]:
            for _ in range(2 if modele.endswith(("350", "30")) else 1):
                k += 1
                eid = f"{prefix}-{modele.replace('-', '')}-{k:03d}"
                equipements.append((eid, famille, modele, random.choice(["Atelier A", "Atelier B"]),
                                    f"202{random.randint(1, 4)}-0{random.randint(1, 9)}-15", fam["groupe"]))
    con.executemany("INSERT INTO equipements VALUES (?,?,?,?,?,?)", equipements)

    techs = {"pompe": ["tech_p01", "tech_p02"], "compresseur": ["tech_c01", "tech_c02"]}
    rows, start = [], date(2025, 6, 1)
    for eid, famille, modele, *_ in equipements:
        fam = FAMILLES[famille]
        for _ in range(random.randint(4, 7)):
            d = start + timedelta(days=random.randint(0, 500))
            if random.random() < 0.5:
                code, lib, *_rest, action = random.choice(fam["alarmes"])
                rows.append((eid, d.isoformat(), "corrective", code, f"Alarme {code} ({lib}). {action}",
                             random.choice(techs[famille]), round(random.uniform(0.5, 4), 1)))
            else:
                pid, titre, *_r = random.choice(fam["procedures"])
                rows.append((eid, d.isoformat(), "preventive", None, f"{pid} {titre} réalisée.",
                             random.choice(techs[famille]), round(random.uniform(0.5, 3), 1)))
    rows.sort(key=lambda r: r[1])
    con.executemany("INSERT INTO interventions (equipment_id,date,type,code_alarme,description,technicien,"
                    "duree_h) VALUES (?,?,?,?,?,?,?)", rows)
    con.executemany("INSERT INTO tickets VALUES (?,?,?,?,?,?,?,?)", [
        ("TCK-0001", "POMPE-HF350-002", "Fuite garniture mécanique", "Alarme E05 récurrente, fuite visible.",
         "ouvert", "tech_p01", "2026-10-20", "seed-1"),
        ("TCK-0002", "COMP-AC30-002", "Purgeur bloqué", "Alarme A07 depuis 2 jours.",
         "ouvert", "tech_c02", "2026-10-28", "seed-2"),
        ("TCK-0003", "POMPE-HF200-001", "Remplacement crépine", "Crépine déformée lors du nettoyage.",
         "ferme", "tech_p02", "2026-08-02", "seed-3"),
    ])
    con.commit()
    con.close()
    return len(equipements), len(rows)


def utilisateurs():
    users = [
        {"user_id": "tech_p01", "nom": "Technicien Pompes 1", "groupes": ["tech_pompes"]},
        {"user_id": "tech_p02", "nom": "Technicien Pompes 2", "groupes": ["tech_pompes"]},
        {"user_id": "tech_c01", "nom": "Technicien Compresseurs 1", "groupes": ["tech_compresseurs"]},
        {"user_id": "tech_c02", "nom": "Technicien Compresseurs 2", "groupes": ["tech_compresseurs"]},
        {"user_id": "resp_01", "nom": "Responsable Maintenance", "groupes": ["responsable"]},
    ]
    (ROOT / "users.json").write_text(json.dumps(users, ensure_ascii=False, indent=2), encoding="utf-8")


# --------------------------------------------------------------------------- main

def main():
    for p in (CORPUS, FIGURES, PHOTOS, MCP):
        p.mkdir(parents=True, exist_ok=True)

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
