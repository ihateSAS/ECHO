#!/usr/bin/env python3


from __future__ import annotations

import math
from pathlib import Path

from reportlab.lib import colors
from reportlab.lib.enums import TA_CENTER, TA_LEFT
from reportlab.lib.pagesizes import letter
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
from reportlab.lib.units import inch
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont
from reportlab.platypus import (
    BaseDocTemplate,
    Flowable,
    Frame,
    Image,
    KeepTogether,
    PageBreak,
    PageTemplate,
    Paragraph,
    Spacer,
    Table,
    TableStyle,
)


REPO = Path(__file__).resolve().parents[1]
OUTPUT = REPO / "output" / "pdf" / "binauralcot_methods_and_math_guide.pdf"
FIGURES = REPO / "paper" / "figures"
FONT_DIR = REPO / ".venv" / "lib" / "python3.13" / "site-packages" / "matplotlib" / "mpl-data" / "fonts" / "ttf"

NAVY = colors.HexColor("#13263D")
BLUE = colors.HexColor("#3569B8")
PALE_BLUE = colors.HexColor("#EAF1FA")
GOLD = colors.HexColor("#D9A72E")
PALE_GOLD = colors.HexColor("#FCF5DF")
GREEN = colors.HexColor("#287A4D")
PALE_GREEN = colors.HexColor("#EAF5EE")
RED = colors.HexColor("#A5413E")
PALE_RED = colors.HexColor("#FAECEB")
INK = colors.HexColor("#20252B")
MUTED = colors.HexColor("#5C6670")
RULE = colors.HexColor("#CBD2D9")
LIGHT = colors.HexColor("#F5F7F9")


def register_fonts() -> None:
    pdfmetrics.registerFont(TTFont("CMR", FONT_DIR / "cmr10.ttf"))
    pdfmetrics.registerFont(TTFont("CMB", FONT_DIR / "cmb10.ttf"))
    pdfmetrics.registerFont(TTFont("CMTT", FONT_DIR / "cmtt10.ttf"))
    pdfmetrics.registerFontFamily("ComputerModern", normal="CMR", bold="CMB")


register_fonts()


styles = getSampleStyleSheet()
TITLE = ParagraphStyle(
    "GuideTitle",
    fontName="CMB",
    fontSize=28,
    leading=32,
    textColor=NAVY,
    alignment=TA_CENTER,
    spaceAfter=12,
)
SUBTITLE = ParagraphStyle(
    "GuideSubtitle",
    fontName="CMR",
    fontSize=14,
    leading=19,
    textColor=MUTED,
    alignment=TA_CENTER,
    spaceAfter=18,
)
H1 = ParagraphStyle(
    "H1",
    fontName="CMB",
    fontSize=18,
    leading=22,
    textColor=NAVY,
    spaceBefore=4,
    spaceAfter=10,
    keepWithNext=True,
)
H2 = ParagraphStyle(
    "H2",
    fontName="CMB",
    fontSize=13,
    leading=16,
    textColor=BLUE,
    spaceBefore=9,
    spaceAfter=5,
    keepWithNext=True,
)
BODY = ParagraphStyle(
    "Body",
    fontName="CMR",
    fontSize=10.4,
    leading=14.2,
    textColor=INK,
    spaceAfter=7,
)
SMALL = ParagraphStyle(
    "Small",
    parent=BODY,
    fontSize=8.7,
    leading=11.5,
    textColor=MUTED,
)
CAPTION = ParagraphStyle(
    "Caption",
    parent=SMALL,
    fontSize=8.4,
    leading=10.8,
    alignment=TA_LEFT,
    spaceBefore=4,
    spaceAfter=10,
)
EQ = ParagraphStyle(
    "Equation",
    fontName="CMTT",
    fontSize=10.3,
    leading=15,
    alignment=TA_CENTER,
    textColor=NAVY,
)
BOX_BODY = ParagraphStyle(
    "BoxBody",
    parent=BODY,
    fontSize=9.5,
    leading=12.8,
    spaceAfter=0,
)
TABLE_HEAD = ParagraphStyle(
    "TableHead",
    fontName="CMB",
    fontSize=8.5,
    leading=10,
    textColor=colors.white,
    alignment=TA_CENTER,
)
TABLE_CELL = ParagraphStyle(
    "TableCell",
    fontName="CMR",
    fontSize=8.3,
    leading=10.2,
    textColor=INK,
    alignment=TA_CENTER,
)
TABLE_LEFT = ParagraphStyle(
    "TableLeft",
    parent=TABLE_CELL,
    alignment=TA_LEFT,
)


class SectionRule(Flowable):
    def __init__(self, width: float):
        super().__init__()
        self.width = width
        self.height = 6

    def draw(self) -> None:
        self.canv.setStrokeColor(GOLD)
        self.canv.setLineWidth(2.2)
        self.canv.line(0, 3, self.width, 3)


def P(text: str, style: ParagraphStyle = BODY) -> Paragraph:
    return Paragraph(text, style)


def section(title: str, width: float) -> list[Flowable]:
    return [P(title, H1), SectionRule(width), Spacer(1, 5)]


def subhead(title: str) -> Paragraph:
    return P(title, H2)


def callout(title: str, text: str, *, kind: str = "blue") -> Table:
    palette = {
        "blue": (PALE_BLUE, BLUE),
        "gold": (PALE_GOLD, GOLD),
        "green": (PALE_GREEN, GREEN),
        "red": (PALE_RED, RED),
    }
    fill, edge = palette[kind]
    content = P(f"<font name='CMB'>{title}</font><br/>{text}", BOX_BODY)
    table = Table([[content]], colWidths=[6.95 * inch])
    table.setStyle(
        TableStyle(
            [
                ("BACKGROUND", (0, 0), (-1, -1), fill),
                ("BOX", (0, 0), (-1, -1), 0.9, edge),
                ("LEFTPADDING", (0, 0), (-1, -1), 10),
                ("RIGHTPADDING", (0, 0), (-1, -1), 10),
                ("TOPPADDING", (0, 0), (-1, -1), 8),
                ("BOTTOMPADDING", (0, 0), (-1, -1), 8),
            ]
        )
    )
    return table


def equation(text: str) -> Table:
    table = Table([[P(text, EQ)]], colWidths=[6.55 * inch])
    table.setStyle(
        TableStyle(
            [
                ("BACKGROUND", (0, 0), (-1, -1), LIGHT),
                ("BOX", (0, 0), (-1, -1), 0.7, RULE),
                ("LEFTPADDING", (0, 0), (-1, -1), 8),
                ("RIGHTPADDING", (0, 0), (-1, -1), 8),
                ("TOPPADDING", (0, 0), (-1, -1), 7),
                ("BOTTOMPADDING", (0, 0), (-1, -1), 7),
            ]
        )
    )
    return table


def bullet(text: str) -> Paragraph:
    return Paragraph(f"- {text}", ParagraphStyle("Bullet", parent=BODY, leftIndent=15, firstLineIndent=-9, spaceAfter=4))


def data_table(rows: list[list[str]], widths: list[float]) -> Table:
    cooked: list[list[Paragraph]] = []
    for row_index, row in enumerate(rows):
        cooked.append(
            [
                P(cell, TABLE_HEAD if row_index == 0 else (TABLE_LEFT if col_index == 0 else TABLE_CELL))
                for col_index, cell in enumerate(row)
            ]
        )
    table = Table(cooked, colWidths=widths, repeatRows=1, hAlign="LEFT")
    table.setStyle(
        TableStyle(
            [
                ("BACKGROUND", (0, 0), (-1, 0), NAVY),
                ("BACKGROUND", (0, 1), (-1, -1), colors.white),
                ("ROWBACKGROUNDS", (0, 1), (-1, -1), [colors.white, LIGHT]),
                ("BOX", (0, 0), (-1, -1), 0.7, RULE),
                ("INNERGRID", (0, 0), (-1, -1), 0.35, RULE),
                ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
                ("LEFTPADDING", (0, 0), (-1, -1), 5),
                ("RIGHTPADDING", (0, 0), (-1, -1), 5),
                ("TOPPADDING", (0, 0), (-1, -1), 5),
                ("BOTTOMPADDING", (0, 0), (-1, -1), 5),
            ]
        )
    )
    return table


def fitted_image(path: Path, max_width: float, max_height: float) -> Image:
    image = Image(str(path))
    scale = min(max_width / image.imageWidth, max_height / image.imageHeight)
    image.drawWidth = image.imageWidth * scale
    image.drawHeight = image.imageHeight * scale
    image.hAlign = "CENTER"
    return image


def page_decor(canvas, doc) -> None:
    page = canvas.getPageNumber()
    canvas.saveState()
    if page > 1:
        canvas.setFont("CMR", 8)
        canvas.setFillColor(MUTED)
        canvas.drawString(doc.leftMargin, letter[1] - 0.38 * inch, "BinauralCoT - Methods and Math Guide")
        canvas.setStrokeColor(RULE)
        canvas.setLineWidth(0.5)
        canvas.line(doc.leftMargin, letter[1] - 0.44 * inch, letter[0] - doc.rightMargin, letter[1] - 0.44 * inch)
    canvas.setFont("CMR", 8)
    canvas.setFillColor(MUTED)
    canvas.drawCentredString(letter[0] / 2, 0.35 * inch, str(page))
    canvas.restoreState()


def build_story(doc_width: float) -> list[Flowable]:
    story: list[Flowable] = []

    story.extend(
        [
            Spacer(1, 0.60 * inch),
            P("BinauralCoT", TITLE),
            P("Methods and Math - A Guide for Writing the Paper Yourself", SUBTITLE),
            Spacer(1, 0.08 * inch),
            callout(
                "The project in one sentence",
                "BinauralCoT holds a spatial-localization problem fixed and changes only how the physically sufficient cue reaches the model, separating cue access from cue-conditioned use.",
                kind="gold",
            ),
            Spacer(1, 0.20 * inch),
            fitted_image(FIGURES / "fig0_framework_and_benchmark.png", 6.9 * inch, 2.8 * inch),
            P(
                "The complete experimental logic: construct verifiable stereo items, present matched cue-text and audio-only views, score the same target, and inspect the gap.",
                CAPTION,
            ),
            Spacer(1, 0.08 * inch),
            P(
                "This guide explains what was done, why each design choice exists, how the equations connect the waveform to the label, and what each experiment can legitimately establish.",
                BODY,
            ),
            Spacer(1, 0.18 * inch),
            P("Prepared from the current project artifacts - August 15, 2026", SMALL),
        ]
    )

    story.append(PageBreak())
    story.extend(section("1. The experimental problem", doc_width))
    story.append(
        P(
            "A model that answers where a sound is located must complete at least two operations. First, it must obtain a spatial cue from the audio. Second, it must use that cue to infer a side or angle. A normal end-to-end score combines both operations, so a wrong answer does not reveal where failure occurred.",
        )
    )
    story.append(subhead("The controlled intervention"))
    story.append(
        data_table(
            [
                ["Element", "Cue-text condition", "Audio-only condition"],
                ["Musical source", "Same held-out URMP excerpt", "Same held-out URMP excerpt"],
                ["Target", "Same side and angle", "Same side and angle"],
                ["Question", "Same localization task", "Same localization task"],
                ["Spatial access", "Measured ILD and law are explicit", "Left and right audio carry the cue"],
                ["What must work", "Cue-conditioned use", "End-to-end cue access and use"],
            ],
            [1.35 * inch, 2.80 * inch, 2.80 * inch],
        )
    )
    story.append(Spacer(1, 8))
    story.append(
        callout(
            "What is manipulated?",
            "Only the practical access path for spatial evidence. The experiment does not change the correct answer between conditions.",
            kind="blue",
        )
    )
    story.append(Spacer(1, 7))
    story.append(
        callout(
            "Careful terminology",
            "Perception and reasoning are useful shorthand. The precise terms are end-to-end cue access and cue-conditioned reasoning, because the audio-only path includes preprocessing, encoding, comparison, projection, alignment, and decoding.",
            kind="green",
        )
    )
    story.append(subhead("Framework versus benchmark"))
    story.append(bullet("BinauralCoT is the diagnostic framework: the matched intervention and interpretation."))
    story.append(bullet("StereoMusicQA is the benchmark: real source signals, controlled rendering, private labels, scoring, and verification."))
    story.append(
        P(
            "A useful sentence for the paper is: StereoMusicQA supplies physically grounded evidence, while BinauralCoT specifies the experiment performed on that evidence.",
        )
    )

    story.append(PageBreak())
    story.extend(section("2. How StereoMusicQA is built", doc_width))
    story.append(
        P(
            "The source material is 149 isolated mono instrument stems from 44 classical chamber pieces in URMP. URMP provides real musical content but not the spatial labels used here. Spatial ground truth is installed by rendering each mono excerpt at a known position.",
        )
    )
    story.append(subhead("Deterministic construction"))
    for text in [
        "Sort the 44 piece identifiers and shuffle them with seed 2026.",
        "Split by whole piece: 31 train, 7 development, and 6 test pieces. No piece crosses splits.",
        "For every stem at least six seconds long, derive a stem-specific seed and draw one six-second excerpt.",
        "Reuse that exact excerpt across all positions and processing conditions, so content does not change with position.",
        "Render at 48 kHz and save accepted audio as 24-bit PCM WAV.",
    ]:
        story.append(bullet(text))
    story.append(subhead("Positions and processing"))
    story.append(
        data_table(
            [
                ["Component", "Setting", "Reason"],
                ["Initial angles", "-25 to +25 degrees in 5-degree steps", "Eleven controlled positions inside the +/-30-degree speaker span"],
                ["Dry", "Amplitude pan only", "Clean invertible reference"],
                ["Compression", "4:1 above -20 dB", "Tests level robustness"],
                ["Equalization", "+6 dB at 1 kHz, Q=1", "Tests spectral processing"],
                ["Mid/side", "Width 1.2", "Changes the effective pan coordinate in a known way"],
                ["Delay widening", "Right channel delayed 300 microseconds", "Adds timing without discarding the recoverable level cue"],
            ],
            [1.25 * inch, 2.15 * inch, 3.55 * inch],
        )
    )
    story.append(Spacer(1, 7))
    story.append(
        P(
            "The generator creates 8,195 candidate renders. Validation rejects non-stereo or non-finite audio, signals below -45 dB RMS, peaks above 0.99, incorrect timing or regime measurements, low coherence when compactness should remain, or any item whose processing-adjusted angle cannot be recovered within 5 degrees.",
        )
    )
    story.append(
        callout(
            "Final single-source benchmark",
            "7,718 accepted items: 5,332 train, 1,341 development, and 1,045 test. The main test set contains 19 stems from six held-out pieces.",
            kind="gold",
        )
    )

    story.append(PageBreak())
    story.extend(section("3. The panning math, step by step", doc_width))
    story.append(
        P(
            "The core idea is to make the answer recoverable from the rendered waveform. Let x[n] be the mono source, let gL and gR be the left and right channel gains, and let theta be the planted azimuth. Positive theta means left; negative theta means right.",
        )
    )
    story.append(subhead("Step 1: turn an angle into a pan coordinate"))
    story.append(equation("d = tan(theta) / tan(30 degrees)"))
    story.append(
        P(
            "The virtual speakers sit at +/-30 degrees. Therefore d=0 is center, d=1 is the left boundary, and d=-1 is the right boundary. The benchmark stays inside the boundary so both channel gains remain nonnegative.",
        )
    )
    story.append(subhead("Step 2: turn the pan coordinate into channel gains"))
    story.append(equation("g_L(raw) = (1 + d) / 2 ; g_R(raw) = (1 - d) / 2"))
    story.append(equation("g_L = g_L(raw) / sqrt(g_L(raw)^2 + g_R(raw)^2)"))
    story.append(equation("g_R = g_R(raw) / sqrt(g_L(raw)^2 + g_R(raw)^2)"))
    story.append(
        P(
            "The final normalization is constant-power scaling. It changes overall level but not the ratio of the left gain to the right gain, so it does not change the spatial answer.",
        )
    )
    story.append(subhead("Step 3: render the stereo waveform"))
    story.append(equation("left[n] = g_L * x[n] ; right[n] = g_R * x[n]"))
    story.append(
        P(
            "Both channels contain the same source waveform at different gains. In the dry regime there is no timing difference. Position is encoded in their relative level.",
        )
    )
    story.append(subhead("Step 4: measure inter-channel level difference"))
    story.append(equation("ILD = 20 * log10(g_L / g_R)    [dB]"))
    story.append(
        P(
            "ILD is positive when the left channel is stronger, negative when the right channel is stronger, and zero at center. The benchmark measures this from the finished audio rather than trusting rendering metadata alone.",
        )
    )

    story.append(PageBreak())
    story.extend(section("4. Inverting the cue back into an answer", doc_width))
    story.append(
        P(
            "The model is shown the following inverse in the cue-text condition. These equations recover the angle from the measured ILD.",
        )
    )
    story.append(subhead("Step 1: recover the gain ratio"))
    story.append(equation("r = 10^(ILD / 20) = g_L / g_R"))
    story.append(
        P(
            "The logarithmic decibel measurement is converted back to a linear amplitude ratio. For example, 6 dB means the left amplitude is about 1.995 times the right amplitude.",
        )
    )
    story.append(subhead("Step 2: recover the normalized pan coordinate"))
    story.append(equation("d = (r - 1) / (r + 1)"))
    story.append(
        P(
            "This follows from r=(1+d)/(1-d). It maps every positive gain ratio to a bounded coordinate between -1 and 1.",
        )
    )
    story.append(subhead("Step 3: recover the azimuth"))
    story.append(equation("theta = atan(d * tan(30 degrees))"))
    story.append(
        P(
            "The arctangent reverses the tangent-law coordinate. The sign supplies side; the magnitude supplies distance from center within the studio pan span.",
        )
    )
    r6 = 10 ** (6 / 20)
    d6 = (r6 - 1) / (r6 + 1)
    theta6 = math.degrees(math.atan(d6 * math.tan(math.radians(30))))
    story.append(subhead("Worked example: ILD = +6.0 dB"))
    story.append(
        data_table(
            [
                ["Quantity", "Calculation", "Value"],
                ["Ratio", "r = 10^(6/20)", f"{r6:.4f}"],
                ["Pan coordinate", "d = (r-1)/(r+1)", f"{d6:.4f}"],
                ["Angle", "atan(d*tan(30 degrees))", f"+{theta6:.2f} degrees"],
                ["Interpretation", "Positive sign", "10.86 degrees left"],
            ],
            [1.35 * inch, 3.25 * inch, 2.35 * inch],
        )
    )
    story.append(Spacer(1, 7))
    story.append(
        callout(
            "Why this is physics-grounded",
            "The label is not merely declared by the renderer. The finished waveform is measured, the cue is recomputed, and the inverse law must recover the processing-adjusted answer.",
            kind="green",
        )
    )

    story.append(PageBreak())
    story.extend(section("5. How each benchmark condition is delivered", doc_width))
    story.append(subhead("Cue-text condition"))
    for text in [
        "Measure ILD on the 48 kHz stereo render.",
        "Downmix the audio to mono so the model does not need to infer position from its audio input.",
        "State the measured ILD, louder channel, sign convention, and inverse panning law in text.",
        "Ask for side and azimuth. This measures best-case use of an explicit symbolic cue.",
    ]:
        story.append(bullet(text))
    story.append(subhead("Audio-only condition"))
    for text in [
        "Supply the left waveform as Audio 1 and the right waveform as Audio 2.",
        "Identify channel order but provide no measured ILD or conversion law.",
        "Ask the same side-and-angle question with the same target.",
        "The model must preserve, compare, align, and use the channel relationship through its audio-language interface.",
    ]:
        story.append(bullet(text))
    story.append(subhead("Model-specific delivery"))
    story.append(
        P(
            "Qwen2-Audio independently resamples both clips to 16 kHz and decodes greedily with at most 200 new tokens. Gemini Flash-Lite and GPT-audio receive 48 kHz PCM WAV inputs through their hosted APIs. Each item receives one completion per condition; there is no self-consistency selection.",
        )
    )
    story.append(
        callout(
            "Important limitation",
            "Two separately identified clips are a controlled interface, not native stereo ingestion. Failure can arise in channel comparison, temporal alignment, projection, prompting, or decoding. The paper groups these under cue access instead of claiming pure sensory loss.",
            kind="red",
        )
    )

    story.append(PageBreak())
    story.extend(section("6. Scoring and the premise-grounded verifier", doc_width))
    story.append(
        data_table(
            [
                ["Outcome", "Definition", "What it tells us"],
                ["Side accuracy", "Correct left, right, or center", "Qualitative direction"],
                ["Within-5-degree accuracy", "Angular error no more than 5 degrees", "Useful quantitative localization"],
                ["Mean absolute error", "Mean absolute difference between predicted and true angle", "Magnitude of numerical error"],
                ["Coverage", "Fraction with a parseable angle", "Whether missing outputs hide failure"],
                ["Derivation faithfulness", "Written steps pass premise and physics checks", "Consistency of the stated explanation"],
            ],
            [1.35 * inch, 2.70 * inch, 2.90 * inch],
        )
    )
    story.append(Spacer(1, 8))
    story.append(subhead("Why the verifier needed a premise check"))
    story.append(
        P(
            "The first verifier compared isolated claims with stored ground truth. That could accept a derivation whose final answer was near the label even when its intermediate arithmetic used a different cue. The corrected verifier connects the model's written premise to its written intermediate value.",
        )
    )
    story.append(equation("If the model writes r = 10^(x/20) = y, recompute y from its stated x."))
    story.append(subhead("Concrete off-grid example"))
    story.append(
        data_table(
            [
                ["Statement", "Value", "Interpretation"],
                ["Cue written by model", "ILD = -14.8 dB", "A new off-grid cue"],
                ["Ratio written by model", "r = 0.139", "Incorrect for -14.8 dB"],
                ["Correct ratio", "10^(-14.8/20) = 0.182", "Must be used by a premise-consistent derivation"],
                ["Model's final answer", "-23.6 degrees", "Matches a trained cue instead of the stated cue"],
                ["Correct answer", "about -21.8 degrees", "Recovered from the stated premise"],
            ],
            [1.55 * inch, 2.20 * inch, 3.20 * inch],
        )
    )
    story.append(Spacer(1, 7))
    story.append(
        callout(
            "What faithfulness means here",
            "The verifier audits the derivation the model chose to write. It does not reveal the model's hidden causal reasoning process.",
            kind="gold",
        )
    )

    story.append(PageBreak())
    story.extend(section("7. Why the statistics resample stems", doc_width))
    story.append(
        P(
            "The 1,045 test items are transformations of only 19 source stems. Renders of the same stem share timbre, performance, and recording properties, so treating all 1,045 rows as independent would make uncertainty look too small.",
        )
    )
    story.append(subhead("Paired cluster bootstrap"))
    for text in [
        "Use the 19 held-out stems as clusters.",
        "Draw 19 stem identifiers with replacement.",
        "For every selected stem, include its complete set of rendered items. A stem drawn twice contributes twice.",
        "Keep cue-text and audio-only correctness paired on every item.",
        "Compute cue-text accuracy, audio-only accuracy, and their within-model difference.",
        "Repeat 20,000 times with seed 20,260,814.",
        "Use the empirical 2.5th and 97.5th percentiles as the 95% interval.",
    ]:
        story.append(bullet(text))
    story.append(equation("gap^(b) = accuracy_cue^(b) - accuracy_audio^(b)"))
    story.append(
        P(
            "Because each bootstrap replicate uses the same sampled stems for both conditions, shared easy or difficult recordings cancel in the difference instead of inflating uncertainty.",
        )
    )
    story.append(
        data_table(
            [
                ["Model", "Cue-text", "Audio-only", "Paired gap and 95% interval"],
                ["Qwen2-Audio", "71.9%", "28.5%", "+43.3 [35.9, 50.9] points"],
                ["Gemini Flash-Lite", "93.6%", "36.6%", "+57.0 [54.0, 60.0] points"],
                ["GPT-audio", "90.4%", "42.0%", "+48.4 [46.2, 50.8] points"],
            ],
            [1.55 * inch, 1.25 * inch, 1.25 * inch, 2.90 * inch],
        )
    )

    story.append(PageBreak())
    story.extend(section("8. The main statistical result", doc_width))
    story.append(fitted_image(FIGURES / "fig1_reasoning_vs_perception.png", 6.95 * inch, 3.15 * inch))
    story.append(
        P(
            "Panels (a) and (b) show the same side-accuracy metric. Panel (c) shows the paired cue-text advantage. Error bars are the stem-cluster bootstrap intervals, not item-level binomial intervals.",
            CAPTION,
        )
    )
    story.append(
        callout(
            "Central empirical finding",
            "All three systems are 43.3 to 57.0 percentage points more accurate when the cue is explicit. Every paired interval remains above zero, while every audio-only point estimate remains below the 45.5% tied-majority baseline.",
            kind="green",
        )
    )
    story.append(subhead("Why side accuracy is not full localization"))
    story.append(
        data_table(
            [
                ["Model", "Cue-text side", "Cue-text within 5 degrees", "Cue-text MAE"],
                ["Qwen2-Audio", "71.9%", "0.4%", "74.27 degrees"],
                ["Gemini Flash-Lite", "93.6%", "44.9%", "7.49 degrees"],
                ["GPT-audio", "90.4%", "80.7%", "5.44 degrees"],
            ],
            [1.65 * inch, 1.65 * inch, 2.05 * inch, 1.60 * inch],
        )
    )
    story.append(Spacer(1, 7))
    story.append(
        P(
            "A model can often use the sign of ILD to choose a side while failing the nonlinear conversion to angle. Boundary substitution is a concrete example: the model repeats the +/-30-degree limit mentioned in the law rather than evaluating the item's expression.",
        )
    )

    story.append(PageBreak())
    story.extend(section("9. The Qwen2-Audio encoder probe", doc_width))
    story.append(
        P(
            "The probe asks whether Qwen2-Audio's frozen encoder has already erased the level cue. It does not ask whether the full model naturally uses that cue.",
        )
    )
    story.append(subhead("Procedure"))
    for text in [
        "Choose ten held-out six-second excerpts using seed 2026.",
        "Render each excerpt at the eleven main benchmark angles, producing 110 observations.",
        "Encode the left and right channels separately and mean-pool valid frames.",
        "Give the probe the explicit difference encoder(left) - encoder(right).",
        "Fit ridge regression with penalty 1.0 and leave one entire excerpt out per fold.",
        "Shuffle angle labels within each excerpt with seed 2027 as a negative control.",
    ]:
        story.append(bullet(text))
    story.append(fitted_image(FIGURES / "fig_appendix_encoder_probe.png", 6.6 * inch, 2.65 * inch))
    story.append(
        P(
            "Azimuth remains highly decodable after encoding when the comparison is supplied externally. The shuffled control fails, showing that high dimensionality alone does not explain the result.",
            CAPTION,
        )
    )
    story.append(
        callout(
            "Supported interpretation",
            "The cue survives in a linearly decodable form after Qwen's encoder. The bottleneck may therefore involve forming the channel difference, aligning the two sequences, projecting the comparison, or decoding it into language.",
            kind="blue",
        )
    )
    story.append(Spacer(1, 7))
    story.append(
        callout(
            "Not supported",
            "The probe does not prove that Qwen itself computes encoder(left) - encoder(right), that every spatial feature survives, or that Gemini and GPT-audio fail for the same mechanism.",
            kind="red",
        )
    )

    story.append(PageBreak())
    story.extend(section("10. Off-grid auditing and benchmark hardening", doc_width))
    story.append(
        P(
            "The first cue-text fine-tune reached 100% within-5-degree accuracy on held-out items. However, those items reused the same small set of displayed cue-answer relationships as training. There were only 21 distinct rows to memorize.",
        )
    )
    story.append(subhead("The off-grid test"))
    for text in [
        "Construct 190 items at ten positions absent from the original grid.",
        "Check whether answers land on an original trained angle.",
        "Check whether the written intermediate ratio follows the new stated ILD.",
        "Re-run the corrected premise-grounded verifier.",
    ]:
        story.append(bullet(text))
    story.append(
        data_table(
            [
                ["Finding", "Original adapter"],
                ["Answers on an original trained angle", "190/190"],
                ["Transcripts substituting a trained cue", "189/190"],
                ["Within-5-degree accuracy", "80.0%"],
                ["Mean absolute error", "4.81 degrees"],
                ["Premise-grounded faithfulness", "0.5%"],
            ],
            [3.90 * inch, 3.05 * inch],
        )
    )
    story.append(Spacer(1, 8))
    story.append(subhead("Dense repair"))
    story.append(
        P(
            "The repair appends 3,000 continuously sampled cue-answer pairs generated with seed 2026. Off-grid evaluation angles are excluded with a 0.2-degree margin. The cue is converted by the exact inverse law, and the target is generated forward from the displayed premise.",
        )
    )
    story.append(fitted_image(FIGURES / "fig2_lookup_table_before_after.png", 6.65 * inch, 2.35 * inch))
    story.append(
        P(
            "After dense augmentation, off-grid within-5-degree accuracy is 100%, MAE is 0.24 degrees, verifier faithfulness is 100%, and no answer lands on the original 21-angle grid.",
            CAPTION,
        )
    )
    story.append(
        callout(
            "Correct conclusion",
            "Dense augmentation removes the demonstrated 21-row shortcut. It does not prove symbolic execution, because dense interpolation remains a complete alternative explanation.",
            kind="gold",
        )
    )

    story.append(PageBreak())
    story.extend(section("11. Preliminary mitigation: dithering", doc_width))
    story.append(
        P(
            "Dithering adds independent Gaussian noise to the left and right channels before Qwen2-Audio processes them. The intended mechanism is to create inter-channel variance that survives normalization.",
        )
    )
    story.append(equation("left' = left + 0.05 * N_L ; right' = right + 0.05 * N_R"))
    story.append(equation("N_L and N_R are independent samples from N(0,1)"))
    story.append(
        data_table(
            [
                ["Configuration", "Side accuracy", "Interpretation"],
                ["Zero-shot audio-only", "28.5%", "No robust localization"],
                ["Fine-tune without dither", "44.4%", "Answer collapse near baseline"],
                ["Dither plus encoder Q/K/V LoRA", "75.6%, then 77.1%", "Replicated binary directional signal"],
                ["Fully unfrozen encoder, 1 epoch", "45.5%", "Constant-answer collapse"],
                ["Fully unfrozen encoder, 3 epochs", "46.4%", "Two-template collapse"],
            ],
            [2.70 * inch, 1.35 * inch, 2.90 * inch],
        )
    )
    story.append(Spacer(1, 8))
    story.append(
        P(
            "The positive runs still emit only a few fixed templates, fail center, misidentify the instrument, and do not recover angle magnitude. Dithering is therefore an existence result for recoverable side information, not a solution to the benchmark.",
        )
    )
    story.append(
        callout(
            "Paper role",
            "Keep dithering in preliminary results or future work. The main contribution is the diagnostic benchmark and the failure it reveals; a robust mitigation can become a separate paper.",
            kind="blue",
        )
    )

    story.append(PageBreak())
    story.extend(section("12. What the methods establish", doc_width))
    story.append(subhead("Claims supported by the design"))
    for text in [
        "The same localization problem is much easier for all three models when its sufficient level cue is explicit.",
        "The audio-to-language path does not make that cue reliably usable under the tested two-input interface.",
        "Qualitative side, quantitative angle, and premise-consistent derivation are distinct outcomes.",
        "Qwen's frozen encoder retains a linearly decodable level relationship when external channel differencing is supplied.",
        "Held-out audio can conceal memorized physical relationships; off-grid cue evaluation is necessary.",
    ]:
        story.append(bullet(text))
    story.append(subhead("Claims not supported"))
    for text in [
        "That every audio model has the same bottleneck.",
        "That one exact Qwen layer causes the end-to-end failure.",
        "That amplitude-panned studio stereo is natural individualized binaural hearing.",
        "That written chain-of-thought reveals hidden cognition.",
        "That dense repair proves execution of the symbolic law.",
        "That dithering solves exact spatial localization.",
    ]:
        story.append(bullet(text))
    story.append(Spacer(1, 7))
    story.append(
        callout(
            "The paper's central methodological contribution",
            "Ordinary localization accuracy hides whether a model failed to obtain the cue or failed to use it. BinauralCoT makes that distinction experimentally observable.",
            kind="gold",
        )
    )
    story.append(subhead("Five questions to answer while writing"))
    for text in [
        "What did this experiment hold fixed?",
        "What single variable did it change?",
        "Why is the answer recoverable from the waveform?",
        "Which interpretation follows from the result, and which stronger interpretation does not?",
        "What shortcut or dependency would make the number misleading?",
    ]:
        story.append(bullet(text))
    story.append(
        P(
            "If every methods paragraph answers one of those questions, the paper will read as a chain of evidence rather than a list of implementation details.",
        )
    )

    return story


def main() -> None:
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    page_width, page_height = letter
    left = right = 0.62 * inch
    top = 0.58 * inch
    bottom = 0.55 * inch
    doc = BaseDocTemplate(
        str(OUTPUT),
        pagesize=letter,
        leftMargin=left,
        rightMargin=right,
        topMargin=top,
        bottomMargin=bottom,
        title="BinauralCoT Methods and Math Guide",
        author="BinauralCoT project",
        subject="Teaching guide to the benchmark, matched intervention, physics, verification, statistics, and audits",
    )
    frame = Frame(left, bottom, page_width - left - right, page_height - top - bottom, id="normal")
    doc.addPageTemplates([PageTemplate(id="guide", frames=[frame], onPage=page_decor)])
    doc.build(build_story(doc.width))
    print(OUTPUT)


if __name__ == "__main__":
    main()
