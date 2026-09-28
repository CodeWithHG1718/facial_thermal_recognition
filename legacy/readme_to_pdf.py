"""
readme_to_pdf.py
================
Generates a professional, first-person, humanized PDF from the pipeline README.
Output: c:\\Users\\arpit\\OneDrive\\Desktop\\dataset\\Pipeline_README.pdf
"""

from reportlab.lib.pagesizes import A4
from reportlab.lib import colors
from reportlab.lib.units import cm
from reportlab.lib.styles import ParagraphStyle
from reportlab.lib.enums import TA_LEFT, TA_CENTER, TA_JUSTIFY
from reportlab.platypus import (
    SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle,
    HRFlowable, PageBreak, KeepTogether
)
from reportlab.pdfgen import canvas
from reportlab.lib.colors import HexColor

# ── Page / colour config ─────────────────────────────────────────────────────
PAGE_W, PAGE_H = A4
L_MARGIN = R_MARGIN = 2.0 * cm
USABLE = PAGE_W - L_MARGIN - R_MARGIN   # ~481 pts

C_DARK   = HexColor("#0F172A")
C_ACCENT = HexColor("#2563EB")
C_GREEN  = HexColor("#059669")
C_AMBER  = HexColor("#D97706")
C_NAVY   = HexColor("#1E3A5F")
C_PURPLE = HexColor("#7C3AED")
C_CYAN   = HexColor("#0891B2")
C_ALT    = HexColor("#EEF2FF")
C_LTBLUE = HexColor("#EFF6FF")
C_LTYELL = HexColor("#FFFBEB")
C_GREY   = HexColor("#F1F5F9")
C_BORDER = HexColor("#CBD5E1")
C_MID    = HexColor("#64748B")
C_BODY   = HexColor("#1E293B")
WHITE    = colors.white

OUTPUT = r"c:\Users\arpit\OneDrive\Desktop\dataset\Pipeline_README.pdf"


# ── Page numbering ────────────────────────────────────────────────────────────
class NumberedCanvas(canvas.Canvas):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._pages = []

    def showPage(self):
        self._pages.append(dict(self.__dict__))
        self._startPage()

    def save(self):
        total = len(self._pages)
        for state in self._pages:
            self.__dict__.update(state)
            pg = self._pageNumber
            if pg > 1:
                self.saveState()
                self.setFillColor(C_MID)
                self.setFont("Helvetica", 7.5)
                self.drawRightString(PAGE_W - 2*cm, 1.1*cm,
                                     f"Page {pg} of {total}")
                self.drawString(2*cm, 1.1*cm,
                                "Thermal Breathing Pipeline  |  Sep 2026")
                self.setStrokeColor(C_BORDER)
                self.setLineWidth(0.4)
                self.line(2*cm, 1.4*cm, PAGE_W - 2*cm, 1.4*cm)
                self.restoreState()
            super().showPage()
        super().save()


# ── Styles ────────────────────────────────────────────────────────────────────
def S(name, **kw):
    return ParagraphStyle(name, **kw)

ST = {
    "cover_h1":  S("cover_h1",  fontName="Helvetica-Bold", fontSize=22,
                   textColor=WHITE, alignment=TA_CENTER, leading=30),
    "cover_sub": S("cover_sub", fontName="Helvetica",      fontSize=11,
                   textColor=HexColor("#93C5FD"), alignment=TA_CENTER, leading=16),
    "cover_meta":S("cover_meta",fontName="Helvetica",      fontSize=9,
                   textColor=HexColor("#CBD5E1"), alignment=TA_CENTER, leading=13),
    "h1":        S("h1",        fontName="Helvetica-Bold", fontSize=14,
                   textColor=WHITE, leading=19, spaceBefore=4, spaceAfter=4),
    "h2":        S("h2",        fontName="Helvetica-Bold", fontSize=11,
                   textColor=C_ACCENT, leading=15, spaceBefore=10, spaceAfter=4),
    "h3":        S("h3",        fontName="Helvetica-Bold", fontSize=10,
                   textColor=C_BODY,   leading=14, spaceBefore=7,  spaceAfter=3),
    "body":      S("body",      fontName="Helvetica",      fontSize=9.5,
                   textColor=C_BODY,   alignment=TA_JUSTIFY, leading=15,
                   spaceBefore=3, spaceAfter=3),
    "bullet":    S("bullet",    fontName="Helvetica",      fontSize=9.5,
                   textColor=C_BODY,   leading=14, leftIndent=14,
                   spaceBefore=2, spaceAfter=2),
    "code":      S("code",      fontName="Courier",        fontSize=8.5,
                   textColor=C_DARK,   backColor=C_GREY, leading=13,
                   leftIndent=10, spaceAfter=4),
    "note":      S("note",      fontName="Helvetica",      fontSize=9,
                   textColor=HexColor("#1E3A5F"), leading=13),
    "warn":      S("warn",      fontName="Helvetica",      fontSize=9,
                   textColor=HexColor("#78350F"), leading=13),
    "tbl_hdr":   S("tbl_hdr",  fontName="Helvetica-Bold", fontSize=8.5,
                   textColor=WHITE, alignment=TA_CENTER),
    "tbl_cell":  S("tbl_cell", fontName="Helvetica",      fontSize=8.5,
                   textColor=C_BODY,  alignment=TA_LEFT, leading=12),
    "footer":    S("footer",   fontName="Helvetica",      fontSize=8,
                   textColor=C_MID,   alignment=TA_CENTER, leading=12),
}

def p(text, style="body"):  return Paragraph(text, ST[style])
def sp(h=0.3):               return Spacer(1, h * cm)
def hr(c=C_BORDER, w=0.5):  return HRFlowable(width="100%", thickness=w, color=c)


# ── Banner ────────────────────────────────────────────────────────────────────
def banner(num, title, color=C_ACCENT):
    data = [[p(f"<b>  {num}  {title}</b>", "h1")]]
    t = Table(data, colWidths=[USABLE])
    t.setStyle(TableStyle([
        ("BACKGROUND",    (0,0), (-1,-1), color),
        ("TOPPADDING",    (0,0), (-1,-1), 9),
        ("BOTTOMPADDING", (0,0), (-1,-1), 9),
        ("LEFTPADDING",   (0,0), (-1,-1), 12),
        ("RIGHTPADDING",  (0,0), (-1,-1), 12),
    ]))
    return t


# ── Callout box ───────────────────────────────────────────────────────────────
def callout(text, kind="NOTE"):
    bg  = C_LTBLUE if kind in ("NOTE", "TIP") else C_LTYELL
    bar = C_ACCENT if kind in ("NOTE", "TIP") else C_AMBER
    sty = "note"  if kind in ("NOTE", "TIP") else "warn"
    bar_w = 0.22 * cm
    txt_w = USABLE - bar_w
    data = [[p("", "body"), p(f"<b>{kind}:</b>  {text}", sty)]]
    t = Table(data, colWidths=[bar_w, txt_w])
    t.setStyle(TableStyle([
        ("BACKGROUND",    (0,0), (0,-1), bar),
        ("BACKGROUND",    (1,0), (1,-1), bg),
        ("LEFTPADDING",   (0,0), (0,-1), 0),
        ("RIGHTPADDING",  (0,0), (0,-1), 0),
        ("LEFTPADDING",   (1,0), (1,-1), 10),
        ("RIGHTPADDING",  (1,0), (1,-1), 10),
        ("TOPPADDING",    (0,0), (-1,-1), 8),
        ("BOTTOMPADDING", (0,0), (-1,-1), 8),
        ("BOX",           (0,0), (-1,-1), 0.4, C_BORDER),
    ]))
    return t


# ── Data table ────────────────────────────────────────────────────────────────
def dtable(headers, rows, widths=None):
    if not widths:
        widths = [USABLE / len(headers)] * len(headers)
    data = [[p(h, "tbl_hdr") for h in headers]] + \
           [[p(str(c), "tbl_cell") for c in row] for row in rows]
    t = Table(data, colWidths=widths, repeatRows=1)
    t.setStyle(TableStyle([
        ("BACKGROUND",     (0,0), (-1,0),  C_NAVY),
        ("ROWBACKGROUNDS", (0,1), (-1,-1), [WHITE, C_ALT]),
        ("GRID",           (0,0), (-1,-1), 0.4, C_BORDER),
        ("TOPPADDING",     (0,0), (-1,-1), 5),
        ("BOTTOMPADDING",  (0,0), (-1,-1), 5),
        ("LEFTPADDING",    (0,0), (-1,-1), 7),
        ("RIGHTPADDING",   (0,0), (-1,-1), 7),
        ("VALIGN",         (0,0), (-1,-1), "MIDDLE"),
    ]))
    return t


# ═════════════════════════════════════════════════════════════════════════════
# COVER PAGE
# ═════════════════════════════════════════════════════════════════════════════
def cover():
    el = [sp(1.0)]

    hero = Table(
        [[p("Thermal Breathing Dataset<br/>Pipeline Report", "cover_h1")]],
        colWidths=[USABLE]
    )
    hero.setStyle(TableStyle([
        ("BACKGROUND",    (0,0), (-1,-1), C_DARK),
        ("TOPPADDING",    (0,0), (-1,-1), 32),
        ("BOTTOMPADDING", (0,0), (-1,-1), 32),
        ("ALIGN",         (0,0), (-1,-1), "CENTER"),
    ]))
    el.append(hero)

    sub = Table(
        [[p("Steps 2 and 3: Frozen Frame Removal and Perinasal ROI Detection", "cover_sub")]],
        colWidths=[USABLE]
    )
    sub.setStyle(TableStyle([
        ("BACKGROUND",    (0,0), (-1,-1), C_ACCENT),
        ("TOPPADDING",    (0,0), (-1,-1), 12),
        ("BOTTOMPADDING", (0,0), (-1,-1), 12),
        ("ALIGN",         (0,0), (-1,-1), "CENTER"),
    ]))
    el.append(sub)
    el.append(sp(0.5))

    meta = [
        ["Date",        "September 22, 2026"],
        ["Project",     "Replication of IEEE JTEHM 2023 Breathing Segmentation Paper"],
        ["Dataset",     "5 subjects  -  Joao, anestis, Claudio, Manuel, Jaime"],
        ["Total Frames","15,154 thermal images  (382 x 288 px, 8-bit grayscale PNG)"],
        ["Author",      "Arpit"],
        ["Reference",   "DOI: 10.1109/JTEHM.2023.3295775"],
    ]
    w1, w2 = 3.8*cm, USABLE - 3.8*cm
    md = [[p(f"<b>{r[0]}</b>", "tbl_hdr"), p(r[1], "tbl_cell")] for r in meta]
    mt = Table(md, colWidths=[w1, w2])
    mt.setStyle(TableStyle([
        ("BACKGROUND",     (0,0), (0,-1), C_NAVY),
        ("ROWBACKGROUNDS", (1,0), (1,-1), [HexColor("#F0F4FF"), WHITE]),
        ("GRID",           (0,0), (-1,-1), 0.4, C_BORDER),
        ("TOPPADDING",     (0,0), (-1,-1), 7),
        ("BOTTOMPADDING",  (0,0), (-1,-1), 7),
        ("LEFTPADDING",    (0,0), (-1,-1), 10),
        ("RIGHTPADDING",   (0,0), (-1,-1), 10),
    ]))
    el.append(mt)
    el.append(sp(0.5))
    el.append(hr(C_ACCENT, 1.5))
    el.append(sp(0.3))
    el.append(p(
        "In this report I walk through the two preprocessing steps I performed on the "
        "thermal face dataset that forms the backbone of our breathing monitoring system. "
        "I explain what each step does, why I chose these specific techniques for this "
        "data, the exact results I obtained, and what remains to be done to complete "
        "the full MRF segmentation pipeline."
    ))
    el.append(PageBreak())
    return el


# ═════════════════════════════════════════════════════════════════════════════
# SECTION 1 — Project Background
# ═════════════════════════════════════════════════════════════════════════════
def sec1():
    el = [banner("1", "Project Background", C_ACCENT)]
    el.append(sp(0.4))

    el.append(p(
        "My goal with this project is to replicate the methodology from the 2023 IEEE Journal "
        "of Translational Engineering in Health and Medicine paper titled <i>Breathing-Associated "
        "Facial Region Segmentation for Thermal Camera-Based Indirect Breathing Monitoring</i>. "
        "Instead of detecting nostril openings — which fails at non-frontal camera angles — the "
        "paper's approach exploits a simple physiological fact: when we exhale, warm air exits "
        "through the nose and mouth, periodically heating the surrounding skin. A thermal camera "
        "picks this up as a small but consistent temperature oscillation in the perinasal zone."
    ))
    el.append(sp(0.2))
    el.append(p(
        "What I find elegant about this method is that it requires <b>zero labeled training data</b> "
        "and <b>no manual annotation</b>. A Markov Random Field (MRF) segmentation model automatically "
        "finds the breathing-associated pixels from the data itself, making it fully unsupervised and "
        "adaptable to any camera angle or subject anatomy."
    ))
    el.append(sp(0.3))
    el.append(p("My Dataset", "h2"))
    el.append(p(
        "I am working with five ROS bag files, each containing approximately 2,600 to 3,000 thermal "
        "video frames from a FLIR thermal camera recording a single seated subject at close range. "
        "I previously extracted all frames from the bag files into per-subject PNG folders. Here is "
        "a summary of the raw data I started with:"
    ))
    el.append(sp(0.2))
    el.append(dtable(
        ["Subject", "Original Frames", "Duration (approx)", "Frame Size"],
        [
            ["Joao",    "3,036", "~121 seconds", "382 x 288 px"],
            ["anestis", "2,863", "~115 seconds", "382 x 288 px"],
            ["Claudio", "2,903", "~116 seconds", "382 x 288 px"],
            ["Manuel",  "2,700", "~108 seconds", "382 x 288 px"],
            ["Jaime",   "2,652", "~106 seconds", "382 x 288 px"],
            ["TOTAL",   "15,154","~566 seconds", "8-bit grayscale PNG"],
        ],
        widths=[3.0*cm, 3.5*cm, 3.8*cm, USABLE - 10.3*cm]
    ))
    el.append(PageBreak())
    return el


# ═════════════════════════════════════════════════════════════════════════════
# SECTION 2 — Step 2: Frozen Frame Removal
# ═════════════════════════════════════════════════════════════════════════════
def sec2():
    el = [banner("2", "Step 2 - Frozen Frame Removal", C_GREEN)]
    el.append(sp(0.4))

    el.append(p("What I Discovered", "h2"))
    el.append(p(
        "When I first started analyzing the extracted PNG frames, I noticed something immediately "
        "suspicious: certain blocks of consecutive frames were absolutely identical — not just "
        "similar, but byte-for-byte exact copies of the same image. After investigating further, "
        "I confirmed that across <b>all five subjects</b>, the camera periodically froze for roughly "
        "one second at a time, repeating the last valid captured frame until the data stream resumed. "
        "This is a known hardware artifact caused by <b>camera buffer overflows</b> during USB or "
        "PCIe data transfer in ROS bag recording sessions."
    ))
    el.append(sp(0.25))
    el.append(callout(
        "I found 39 frozen runs across all 5 subjects, totaling 1,186 frozen frames — 7.8% of "
        "the entire dataset. Every single subject is affected, which tells me this was a "
        "systematic issue with the recording setup, not an occasional glitch.",
        "NOTE"
    ))

    el.append(sp(0.3))
    el.append(p("Why I Had to Remove These Frames", "h2"))
    el.append(p(
        "This might seem like a minor issue — after all, 7.8% of frames is not enormous. "
        "But the problem is <i>where</i> these frozen frames appear in the pipeline. "
        "The core of the MRF segmentation relies on computing a <b>rolling standard deviation "
        "over a 100-frame window</b> at each pixel. This standard deviation (the BAF map) is "
        "what reveals the breathing signal — pixels that fluctuate at breathing frequency will "
        "have high variance, while background pixels will not."
    ))
    el.append(sp(0.15))
    el.append(p(
        "A frozen frame has <b>exactly zero variance</b> because it is identical to its neighbours. "
        "If a 31-frame frozen run falls inside a 100-frame computation window, it artificially "
        "reduces that window's computed variance by roughly 31%. This creates false low-variance "
        "zones — regions the algorithm will misclassify as background even if they are right over "
        "the nose and mouth. The result is a corrupted BAFR mask that misses parts of the "
        "breathing region."
    ))

    el.append(sp(0.3))
    el.append(p("How I Detected Them", "h2"))
    el.append(p(
        "The detection logic is straightforward. I compare every frame to the one immediately "
        "before it using a full pixel-by-pixel equality check:"
    ))
    el.append(p("if np.array_equal( frame[t], frame[t-1] ):  # identical = frozen", "code"))
    el.append(p(
        "I flag any run of <b>5 or more</b> consecutive identical frames as a frozen run. "
        "I also remove an extra <b>2 transition frames</b> on either side of each run, because "
        "the frames immediately before and after a freeze often show partial buffer artifacts "
        "that corrupt the signal just as badly as the frozen frames themselves."
    ))

    el.append(sp(0.3))
    el.append(p("Results I Obtained", "h2"))
    el.append(dtable(
        ["Subject", "Total Frames", "Frozen Runs", "Frames Removed", "Valid Frames", "Removed %"],
        [
            ["Joao",    "3,036",  "7",  "224",     "2,812", "7.4%"],
            ["anestis", "2,863",  "8",  "248",     "2,615", "8.7%"],
            ["Claudio", "2,903",  "8",  "245",     "2,658", "8.4%"],
            ["Manuel",  "2,700",  "8",  "229",     "2,471", "8.5%"],
            ["Jaime",   "2,652",  "8",  "240",     "2,412", "9.0%"],
            ["TOTAL",   "15,154", "39", "1,186",   "12,968","7.8%"],
        ],
        widths=[2.5*cm, 2.8*cm, 2.8*cm, 3.2*cm, 2.8*cm, USABLE - 14.1*cm]
    ))
    el.append(sp(0.25))
    el.append(p(
        "I was particularly interested in the regularity of the pattern. Taking Joao as an example, "
        "the seven frozen runs occur at fairly even intervals throughout the 3,036-frame sequence:"
    ))
    el.append(sp(0.15))
    el.append(dtable(
        ["Run", "Start Frame", "End Frame", "Frozen Count", "Approx. Duration"],
        [
            ["1", "345",  "375",  "31 frames", "~1.2 seconds"],
            ["2", "784",  "814",  "31 frames", "~1.2 seconds"],
            ["3", "1190", "1219", "30 frames", "~1.2 seconds"],
            ["4", "1581", "1608", "28 frames", "~1.1 seconds"],
            ["5", "1947", "1973", "27 frames", "~1.1 seconds"],
            ["6", "2339", "2362", "24 frames", "~1.0 seconds"],
            ["7", "2686", "2710", "25 frames", "~1.0 seconds"],
        ],
        widths=[1.5*cm, 3.0*cm, 3.0*cm, 3.0*cm, USABLE - 10.5*cm]
    ))
    el.append(sp(0.2))
    el.append(p(
        "The freezes appear roughly every 400 frames (~16 seconds), and their duration shrinks "
        "slightly over time (from 31 down to 24-25 frames). This pattern is consistent across "
        "all five subjects, confirming that the camera buffer fills up periodically as the "
        "recording session progresses and the system's memory management cycles through."
    ))
    el.append(sp(0.2))
    el.append(p(
        "I saved all valid frames to <b><subject>_clean/</b> folders. The clean folders are the "
        "input for every subsequent pipeline step."
    ))
    el.append(PageBreak())
    return el


# ═════════════════════════════════════════════════════════════════════════════
# SECTION 3 — Step 3: Perinasal ROI Detection
# ═════════════════════════════════════════════════════════════════════════════
def sec3():
    el = [banner("3", "Step 3 - Perinasal ROI Detection and Cropping", C_PURPLE)]
    el.append(sp(0.4))

    el.append(p("What I Needed and Why", "h2"))
    el.append(p(
        "After cleaning the data, I had 12,968 valid thermal frames per subject, each at "
        "382 x 288 pixels. Before I can compute the breathing variance maps, I need to focus "
        "the analysis on the correct part of the image — the <b>perinasal region</b>. "
        "This is the zone around the nostrils, upper lip, and adjacent cheeks where exhaled "
        "air at body temperature (~37°C) periodically warms the skin surface."
    ))
    el.append(sp(0.15))
    el.append(p(
        "If I run the variance computation on the full 382x288 frame, I run into a serious "
        "problem: the background — walls, clothing, furniture — also shows thermal variance "
        "caused by air conditioning drafts and natural thermal gradients. At 382x288 resolution "
        "with the subject seated at a distance, those background regions can be large enough to "
        "completely dominate the BAF variance map and hide the much smaller breathing signal "
        "coming from the perinasal zone. Cropping tightly around the nose and mouth area solves "
        "this and also cuts computation time by about 60%."
    ))

    el.append(sp(0.3))
    el.append(p("How I Detected the Face Region Without a Trained Model", "h2"))
    el.append(p(
        "I want to draw attention to a design decision I made here. Instead of using a pre-trained "
        "face detector (like a Haar cascade or a deep learning model trained on visible-light RGB "
        "images), I wrote a <b>thermal-specific detection method</b> that exploits a simple but "
        "powerful fact about indoor thermal imagery: <i>the human face is almost always the "
        "warmest visible object in the scene</i>."
    ))
    el.append(sp(0.15))
    el.append(p(
        "This matters because RGB-trained detectors don't transfer well to thermal images — they "
        "look for texture and colour patterns that simply do not exist in a grayscale temperature "
        "map. My warm-blob approach works directly in the thermal domain and requires no external "
        "model files. Here is the algorithm I developed:"
    ))
    el.append(sp(0.15))

    steps = [
        ("<b>Threshold at the 75th-percentile brightness:</b> This produces a binary mask of the "
         "top 25% warmest pixels in the frame — which reliably captures the face and any exposed "
         "skin."),
        ("<b>Morphological closing (9x9 ellipse kernel):</b> This fills small gaps in the warm "
         "region caused by glasses, facial hair, or slight temperature variation across the face "
         "surface, joining the face into one solid connected blob."),
        ("<b>Find all contours and pick the largest one:</b> The largest warm connected region "
         "is almost always the face and upper torso. I extract its bounding box."),
        ("<b>Take 35%-95% of the bounding box height:</b> The top 35% of the face bounding box "
         "covers the forehead and eyes — regions that don't participate in breathing. The bottom "
         "5% is typically the chin, which also has minimal breathing signal. The middle 60% "
         "is exactly the nose and mouth area I need."),
        ("<b>Add 8 pixels of padding on all sides:</b> This ensures I don't accidentally clip "
         "the edges of the breathing zone, especially for subjects with wider faces."),
        ("<b>Repeat on 20 evenly-spaced frames and take the median:</b> A single-frame detection "
         "can be thrown off by a blink, a head turn, or a reflection. By sampling 20 frames and "
         "taking the median bounding box coordinates, I get a stable, robust crop region that I "
         "then apply uniformly to every frame in the sequence."),
    ]

    for i, step in enumerate(steps):
        el.append(p(f"&nbsp;&nbsp;&nbsp;{i+1}.  {step}", "bullet"))
        el.append(sp(0.1))

    el.append(sp(0.1))
    el.append(callout(
        "Using the median of 20 sample frames is what makes this approach reliable. "
        "Any single-frame outlier — a blink, a slight head tilt, an arm in frame — "
        "gets averaged out. The resulting ROI box is consistent and accurate across "
        "the entire recording for each subject.",
        "TIP"
    ))

    el.append(sp(0.3))
    el.append(p("Results I Obtained", "h2"))
    el.append(p(
        "The detector correctly localized the perinasal region for all five subjects. "
        "The ROI sizes vary between subjects, which makes sense given natural differences "
        "in face size, seating distance from the camera, and head position. "
        "Each subject also gets their own colour-mapped preview image saved in the dataset "
        "folder so I can visually verify the green ROI rectangle is in the right place."
    ))
    el.append(sp(0.2))
    el.append(dtable(
        ["Subject", "ROI Position", "ROI Size", "Mean Intensity", "Std Dev", "Frames Cropped"],
        [
            ["Joao",    "x=45, y=162", "337 x 124 px", "95.5",  "59.7", "2,812"],
            ["anestis", "x=76, y=150", "281 x 136 px", "111.7", "68.7", "2,615"],
            ["Claudio", "x=62, y=162", "313 x 124 px", "104.3", "64.5", "2,658"],
            ["Manuel",  "x=77, y=149", "276 x 136 px", "117.8", "62.7", "2,471"],
            ["Jaime",   "x=70, y=149", "312 x 136 px", "108.2", "55.5", "2,412"],
        ],
        widths=[2.5*cm, 3.0*cm, 2.8*cm, 2.8*cm, 2.2*cm, USABLE - 13.3*cm]
    ))
    el.append(sp(0.2))
    el.append(p(
        "I am satisfied with these results. The standard deviation values in the 55-69 range "
        "reflect strong thermal contrast within the perinasal region — exactly what I need for "
        "the BAF variance computation to pick up the breathing signal. The mean intensities "
        "vary across subjects (95 to 117), which is expected given different room temperatures "
        "and camera gain settings on recording day. I will normalize these in the per-subject "
        "z-score step later."
    ))
    el.append(PageBreak())
    return el


# ═════════════════════════════════════════════════════════════════════════════
# SECTION 4 — Combined Results
# ═════════════════════════════════════════════════════════════════════════════
def sec4():
    el = [banner("4", "Combined Results Summary", C_CYAN)]
    el.append(sp(0.4))

    el.append(p(
        "Here is the complete picture of what I started with, what I removed, and what I have "
        "available for the next stages of the pipeline:"
    ))
    el.append(sp(0.2))
    el.append(dtable(
        ["Subject", "Original", "Frozen Removed", "Clean Frames", "ROI Crops", "Status"],
        [
            ["Joao",    "3,036",  "224",   "2,812", "2,812", "Complete"],
            ["anestis", "2,863",  "248",   "2,615", "2,615", "Complete"],
            ["Claudio", "2,903",  "245",   "2,658", "2,658", "Complete"],
            ["Manuel",  "2,700",  "229",   "2,471", "2,471", "Complete"],
            ["Jaime",   "2,652",  "240",   "2,412", "2,412", "Complete"],
            ["TOTAL",   "15,154", "1,186", "12,968","12,968","Complete"],
        ],
        widths=[2.5*cm, 2.5*cm, 3.2*cm, 3.0*cm, 3.0*cm, USABLE - 14.2*cm]
    ))
    el.append(sp(0.25))
    el.append(p(
        "Both steps ran fast. Step 2 took about 1.4 minutes for all 5 subjects combined. "
        "Step 3 took about 2 minutes. In total, I went from 15,154 original frames to "
        "<b>12,968 clean, cropped, perinasal ROI images</b> ready for the BAF analysis "
        "in just under 3.5 minutes of processing."
    ))
    el.append(sp(0.3))

    el.append(p("Folder Structure After These Steps", "h2"))
    el.append(p(
        "Each subject now has three folders in the dataset directory:", "body"
    ))
    lines = [
        "<b>Joao/</b>  &nbsp;&nbsp;&nbsp;  Original 8-bit PNG frames from bag extraction (3,036 files)",
        "<b>Joao_clean/</b>  &nbsp;&nbsp;&nbsp;  Frozen-removed valid frames (2,812 files)  [Step 2]",
        "<b>Joao_roi/</b>  &nbsp;&nbsp;&nbsp;  Perinasal ROI crops (2,812 files)  [Step 3]",
        "<b>Joao_roi_preview.png</b>  &nbsp;&nbsp;&nbsp;  Colour thermal preview with green ROI box  [Step 3]",
    ]
    for line in lines:
        el.append(p(f"&nbsp;&nbsp;&nbsp;  {line}", "bullet"))
    el.append(sp(0.2))
    el.append(p(
        "The same pattern repeats for anestis, Claudio, Manuel, and Jaime. From here, "
        "the <b>_roi/ folders are the primary input</b> for all remaining pipeline stages."
    ))
    el.append(PageBreak())
    return el


# ═════════════════════════════════════════════════════════════════════════════
# SECTION 5 — Remaining Pipeline
# ═════════════════════════════════════════════════════════════════════════════
def sec5():
    el = [banner("5", "What Remains to Be Done", C_AMBER)]
    el.append(sp(0.4))

    el.append(p(
        "I have completed the two preprocessing steps. What follows is the core analysis "
        "pipeline that actually produces the BAFR segmentation mask and extracts the "
        "breathing waveform. Here is what I still need to implement:"
    ))
    el.append(sp(0.2))
    el.append(dtable(
        ["Step", "Name", "Input", "Output", "Status"],
        [
            ["0",  "BAG Extraction",         ".bag files",         "Per-subject PNGs",     "Done"],
            ["2",  "Frozen Frame Removal",    "<subject>/",         "<subject>_clean/",     "Done"],
            ["3",  "Perinasal ROI Crop",      "<subject>_clean/",   "<subject>_roi/",       "Done"],
            ["4",  "KLT Motion Tracking",     "ROI frames",         "Stabilised ROI",       "Remaining"],
            ["5",  "BAF Map Computation",     "Stabilised ROI",     "Per-pixel var. maps",  "Remaining"],
            ["6",  "BAP Probability Map",     "BAF maps",           "Normalised [0,1]",     "Remaining"],
            ["7",  "MRF + ICM Segmentation", "BAP maps",           "Binary BAFR mask",     "Remaining"],
            ["8",  "Waveform Extraction",     "BAFR mask + frames", "Breathing signal",     "Remaining"],
            ["9",  "Breathing Rate (BPM)",    "Waveform",           "Breaths per minute",   "Remaining"],
            ["10", "Evaluation",              "BPM estimates",      "Accuracy metrics",     "Remaining"],
        ],
        widths=[1.3*cm, 4.2*cm, 4.0*cm, 4.0*cm, USABLE - 13.5*cm]
    ))
    el.append(sp(0.4))

    remaining = [
        (
            "Step 4 - KLT Optical Flow Motion Compensation", C_GREEN,
            "Even though all five subjects were seated, I can see visible head movement in "
            "their sequences. Even 1-2 pixels of drift per frame, accumulated over 2,500 frames, "
            "creates strong motion-edge artifacts in the variance map that completely swamp the "
            "breathing signal. I will use Kanade-Lucas-Tomasi (KLT) sparse optical flow to track "
            "Shi-Tomasi corner features between consecutive frames, estimate an affine transform, "
            "and warp each ROI frame back to a fixed reference pose. This step is critical — "
            "without it, the BAF map will show the face outline as the highest-variance region "
            "rather than the perinasal breathing zone."
        ),
        (
            "Step 5 - BAF Map: Rolling Per-Pixel Standard Deviation", C_ACCENT,
            "This is the mathematical heart of the pipeline. For each pixel (x, y), I compute "
            "the standard deviation of its intensity values over a rolling 100-frame (4-second) "
            "window:   BAF(x,y,t) = std( T[x,y, t-99 to t] ). The 4-second window is chosen "
            "because normal resting breathing runs at 12-20 breaths per minute, giving a period "
            "of 3-5 seconds. A 4-second window captures exactly one full breathing cycle and "
            "maximally amplifies the thermal oscillation at that frequency. Pixels over the "
            "nose and mouth will show high BAF values; static background pixels will show low values."
        ),
        (
            "Step 6 - BAP Probability Map", C_PURPLE,
            "I normalize each BAF map by its maximum value to produce a spatial probability map: "
            "BAP(x,y,t) = BAF(x,y,t) / max(BAF). The BAP map is in [0,1], making it suitable "
            "as the data term for the MRF segmentation that follows."
        ),
        (
            "Step 7 - MRF + ICM Segmentation (BAFR Mask)", C_CYAN,
            "I will model the segmentation as a 2-class Markov Random Field with an energy "
            "function E(L) = sum E_data(L_i) + lambda * sum E_smooth(L_i,L_j). The data term "
            "penalizes assigning a pixel to BAFR if its BAP value is low, and the smoothness "
            "term (Potts model) penalizes adjacent pixels with different labels. I will minimise "
            "this energy using Iterated Conditional Modes (ICM) over 30 iterations. The output "
            "is a binary BAFR mask: the pixels that are genuinely driven by breathing."
        ),
        (
            "Step 8 - Breathing Waveform Extraction", C_GREEN,
            "With the BAFR mask in hand, I extract the breathing waveform by computing the "
            "spatial mean of the differential thermal signal (frame differences) within the "
            "mask region over time: waveform(t) = mean( delta_T[x,y,t] for (x,y) in BAFR ). "
            "Using frame differences rather than raw values removes slow thermal drift and "
            "greatly improves the signal-to-noise ratio."
        ),
        (
            "Steps 9 and 10 - Rate Estimation and Evaluation", C_AMBER,
            "I will bandpass filter the waveform at 0.1-0.5 Hz (6-30 breaths/min) and apply "
            "peak detection to count breathing cycles. The resulting breaths-per-minute estimate "
            "will be compared against reference measurements if available, using Pearson "
            "correlation, mean absolute error in BPM, and cycle detection accuracy as metrics."
        ),
    ]

    for title, color, desc in remaining:
        block = [
            p(title, "h2"),
            p(desc),
            sp(0.2),
            hr(),
            sp(0.1),
        ]
        el.append(KeepTogether(block))

    el.append(PageBreak())
    return el


# ═════════════════════════════════════════════════════════════════════════════
# SECTION 6 — Scripts Reference
# ═════════════════════════════════════════════════════════════════════════════
def sec6():
    el = [banner("6", "Scripts and How to Run Them", C_NAVY)]
    el.append(sp(0.4))

    el.append(p(
        "All scripts are saved in the dataset folder. Here is a quick reference for "
        "running each one:"
    ))
    el.append(sp(0.2))
    el.append(dtable(
        ["Script", "Purpose", "Command"],
        [
            ["extract_bag_images.py", "Phase 0: Extract PNGs from .bag files",
             "python extract_bag_images.py"],
            ["step2_step3.py",        "Runs Step 2 (frozen removal) from original frames",
             "python step2_step3.py"],
            ["step3_roi.py",          "Step 3 only: ROI detection + crop from _clean/ folders",
             "python step3_roi.py"],
        ],
        widths=[4.2*cm, 6.0*cm, USABLE - 10.2*cm]
    ))
    el.append(sp(0.3))

    el.append(p("Dependencies", "h2"))
    el.append(dtable(
        ["Package", "Version", "Used For"],
        [
            ["numpy",          ">= 1.24", "Array operations, frozen frame equality checks, rolling std dev"],
            ["opencv-python",  ">= 5.0",  "Image I/O, morphological ops, contour detection, colourmap"],
            ["rosbags",        ">= 0.9",  "Phase 0: reading mono16 frames from ROS1 .bag files"],
            ["reportlab",      ">= 5.0",  "Generating this PDF report"],
            ["scipy",          "needed",  "Bandpass filter for Step 9 breathing rate extraction"],
        ],
        widths=[3.5*cm, 2.5*cm, USABLE - 6.0*cm]
    ))
    el.append(sp(0.2))
    el.append(p("python -m pip install numpy opencv-python rosbags reportlab scipy", "code"))
    el.append(sp(0.5))
    el.append(hr(C_ACCENT, 1.5))
    el.append(sp(0.3))
    el.append(p(
        "Report generated: September 22, 2026  |  "
        "Reference: DOI 10.1109/JTEHM.2023.3295775  |  "
        "Dataset: c:\\Users\\arpit\\OneDrive\\Desktop\\dataset\\",
        "footer"
    ))
    return el


# ═════════════════════════════════════════════════════════════════════════════
# BUILD
# ═════════════════════════════════════════════════════════════════════════════
def build():
    doc = SimpleDocTemplate(
        OUTPUT,
        pagesize=A4,
        leftMargin=L_MARGIN, rightMargin=R_MARGIN,
        topMargin=2*cm,      bottomMargin=2.5*cm,
        title="Thermal Breathing Pipeline README",
        author="Arpit",
        subject="Steps 2 and 3 - Frozen Frame Removal and Perinasal ROI",
    )
    story = []
    story += cover()
    story += sec1()
    story += sec2()
    story += sec3()
    story += sec4()
    story += sec5()
    story += sec6()
    doc.build(story, canvasmaker=NumberedCanvas)
    print("DONE: PDF saved to", OUTPUT)


if __name__ == "__main__":
    build()
