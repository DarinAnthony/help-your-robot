"""One-page CS 238 status update as a PDF."""

import sys

from reportlab.lib import colors
from reportlab.lib.enums import TA_CENTER
from reportlab.lib.pagesizes import letter
from reportlab.lib.styles import ParagraphStyle
from reportlab.lib.units import inch
from reportlab.platypus import Paragraph, SimpleDocTemplate, Spacer, Table, TableStyle

OUT = sys.argv[1]
SIZE = 9.4
LEAD = SIZE * 1.24

body = ParagraphStyle("body", fontName="Times-Roman", fontSize=SIZE, leading=LEAD, spaceAfter=2.5)
bullet = ParagraphStyle("bullet", parent=body, leftIndent=10, bulletIndent=1, spaceAfter=1.5)
head = ParagraphStyle("head", fontName="Times-Bold", fontSize=SIZE + 1.4, leading=LEAD + 1, spaceBefore=4, spaceAfter=1.5)
title = ParagraphStyle("title", fontName="Times-Bold", fontSize=14, leading=16, alignment=TA_CENTER, spaceAfter=1)
sub = ParagraphStyle("sub", parent=body, alignment=TA_CENTER, spaceAfter=4)
cell = ParagraphStyle("cell", parent=body, fontSize=SIZE - 0.6, leading=(SIZE - 0.6) * 1.18, spaceAfter=0)
cellb = ParagraphStyle("cellb", parent=cell, fontName="Times-Bold")


def P(text, style=body):
    return Paragraph(text, style)


def B(text):
    return Paragraph(text, bullet, bulletText="•")


def table(rows, widths):
    data = [[P(c, cellb if i == 0 else cell) for c in r] for i, r in enumerate(rows)]
    t = Table(data, colWidths=widths, hAlign="LEFT")
    t.setStyle(TableStyle([
        ("LINEABOVE", (0, 0), (-1, 0), 0.8, colors.black),
        ("LINEBELOW", (0, 0), (-1, 0), 0.5, colors.black),
        ("LINEBELOW", (0, -1), (-1, -1), 0.8, colors.black),
        ("VALIGN", (0, 0), (-1, -1), "TOP"),
        ("TOPPADDING", (0, 0), (-1, -1), 1.2),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 1.2),
        ("LEFTPADDING", (0, 0), (-1, -1), 3),
        ("RIGHTPADDING", (0, 0), (-1, -1), 3),
    ]))
    return t


story = [
    P("When Should a Robot Ask for Help? Status Update", title),
    P("[Team member names] &nbsp;·&nbsp; CS 238 &nbsp;·&nbsp; November 14, 2026", sub),

    P("Problem", head),
    P("A robot doing a manipulation task should sometimes stop and ask a person for help instead of trying again. We "
      "model this as a POMDP. The hidden state is how hard the episode is: <i>easy</i>, <i>recoverable</i> or "
      "<i>infeasible</i>. At each decision point the robot chooses one of three actions: <b>attempt</b> (run the "
      "low-level skill), <b>adjust</b> (re-approach or pick a new grasp), or <b>ask for help</b>, which ends the "
      "episode at a cost. Observations are the outcomes of earlier attempts (a missed grasp, a slip, an unreachable "
      "pose, a drawer that did not move), together with proprioception and camera images. The reward is +10 for "
      "autonomous success, −3 for asking, −0.5 for each failed attempt and a small per-step cost; we will sweep the "
      "help cost. The decision policy plans over a belief whose prior comes from a learned value function, "
      "<i>V(s)</i>&nbsp;=&nbsp;P(success if the robot keeps going), and is updated by Bayes' rule after each attempt."),
    P("<b>Changes since the proposal.</b> We now work in a MuJoCo digital twin of a real kitchen with "
      "system-identified models of two real robots, rather than a generic tabletop. The twin has no cups, so the "
      "objects come from the kitchen's measured object set. We added a second task that isolates "
      "<i>capability</i> uncertainty: a drawer that is normal, stiff, or locked, with nothing visible to tell them "
      "apart."),

    P("Progress", head),
    B("<b>Simulation stack.</b> The kitchen twin is running, with the identified DROID (a Franka arm with a Robotiq "
      "gripper) at a measured pose and the identified TidyBot (a mobile base with an xArm7). Each robot renders "
      "through camera models calibrated to its real cameras."),
    B("<b>Task A: shelf pick-and-place (DROID).</b> A scripted expert moves an object between two spots on a cabinet "
      "shelf 1.5 m up. It makes up to 3 attempts, and each retry changes the approach angle by "
      "15–30°, which is our <i>adjust</i> action. Every attempt is labelled success, unreachable, grasp miss, slip, "
      "place miss or off shelf. We collected 275 episodes (25 per object), keeping the failures."),
    B("<b>Task B: drawer opening (TidyBot).</b> The robot grasps the handle and pulls by driving its base backwards. "
      "We built a <i>hard</i> drawer (slide friction 6.5 N; the normal drawer's is 1–2 N) and a <i>locked</i> one (it stops after 8 mm and holds "
      "200 N). We also found that the simulator re-randomized the drawer's physics every episode and silently "
      "discarded our settings, and fixed it with a hook. We collected 50 episodes."),
]

results = table([
    ["Task", "Condition", "Success", "What it tests"],
    ["A (train)", "wooden ball, tomato, egg, drumstick", "22, 21, 19, 18 of 25", "mostly easy"],
    ["A (train)", "waffle, sponge, duster, banana, tape", "7, 6, 4, 1, 0 of 25", "recoverable or infeasible (too wide for the jaws)"],
    ["A (held out)", "orange (like the training objects)", "21/25", "generalization: should attempt"],
    ["A (held out)", "metal spoon (thin, flat)", "0/25", "should ask early"],
    ["B", "nominal / hard / locked drawer", "20/20, 15/20, 0/10", "same appearance, different capability"],
], [0.8 * inch, 2.3 * inch, 1.25 * inch, 2.75 * inch])
story += [Spacer(1, 2), results, Spacer(1, 3)]

story += [
    P("<b>What we learned.</b> Retries help on round objects: 2–4 extra successes per object came on a second or "
      "third attempt. Retries almost never help on objects wider than the gripper. This is the contrast between "
      "recoverable and infeasible that the planner has to learn. On the drawer, the motion planner usually refuses "
      "before it pulls, because it rehearses the pull in simulation with privileged state; only 1 of the 15 failures "
      "was a pull that actually failed. So our drawer failures are not yet realistic, and fixing this is next."),

    P("Revised timeline", head),
    table([
        ["Week", "Plan"],
        ["Nov 16–22", "Make the drawer failures physical: execute the pull without the planner's rehearsal and record "
                      "force and displacement. Train V0 (logistic regression on hand-built features) and V1 (an "
                      "ensemble on image embeddings and proprioception), calibrate them, and report AUROC, Brier and "
                      "ECE on each split."],
        ["Nov 23–29", "Fit observation models P(outcome | difficulty) from the rollouts. Solve the belief POMDP by "
                      "finite-horizon value iteration. Implement the baselines: never ask, always ask, ask after k "
                      "failures, a threshold on P(success), myopic expected utility, and QMDP."],
        ["Nov 30–Dec 6", "Evaluate closed-loop in the full simulator on the held-out objects and the hard and locked "
                         "drawers. Sweep the help cost and plot success against help rate."],
        ["Dec 7 onward", "Final report. Stretch goal: replace the scripted skill with a fine-tuned π0.5 policy (needs a GPU)."],
    ], [0.95 * inch, 6.15 * inch]),
    Spacer(1, 3),

    P("Feedback we would like", head),
    B("Is a scripted low-level skill acceptable for the main results, with the learned VLA policy as a stretch goal?"),
    B("Should the help cost be a single swept parameter, or grounded in something, such as the human's time?"),
    B("Two tasks or one: should we keep the drawer, or put all our time into the shelf task?"),
    P("<i>AI disclosure:</i> we used AI coding assistants for parts of the simulation code and to help edit this update.",
      ParagraphStyle("ai", parent=body, fontSize=SIZE - 0.8, spaceBefore=2)),
]

doc = SimpleDocTemplate(OUT, pagesize=letter, leftMargin=0.7 * inch, rightMargin=0.7 * inch,
                        topMargin=0.55 * inch, bottomMargin=0.5 * inch,
                        title="When Should a Robot Ask for Help? Status Update", author="CS 238 team")
doc.build(story)
