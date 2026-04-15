"""
Gold-standard query-answer pairs for Sentinel-RAG evaluation.

40 queries across 6 US Army Field Manuals (FM 3-0, FM 3-90, FM 6-0, FM 2-0, FM 5-0, FM 4-0).
Each query is categorized into one of four types:
  - Trap A: "The Overriding Directive" (multi-hop, override/exception across FMs)
  - Trap B: "The Distant Definition" (term used in one FM, defined in another)
  - Trap C: "The Scattered Components" (answer spread across multiple sections/FMs)
  - Control: Single-hop factual queries answerable from one passage

[VERIFY] tags indicate content that should be cross-checked against the actual FM PDFs
by a human annotator. Answers are grounded in actual FM text read during creation.
"""

from __future__ import annotations

from core.data_models import GoldAnnotation, HopCount, QueryCategory

GOLD_ANNOTATIONS: list[GoldAnnotation] = [
    # =========================================================================
    # TRAP A: "The Overriding Directive" (10 queries)
    # A procedure in one FM appears correct, but another FM section overrides it
    # =========================================================================
    GoldAnnotation(
        id="trap_a_01",
        query="During offensive operations, can a brigade combat team commander independently decide to bypass an enemy force encountered during a movement to contact?",
        ground_truth_answer=(
            "Not unconditionally. FM 3-90 (para 1-78 to 1-82) states that after contact, "
            "a unit leader can choose to bypass the enemy as one of the actions on contact. "
            "However, FM 3-90 para 1-80 explicitly states that higher commander approval is "
            "required if the action requires additional resources, is not within the commander's "
            "intent, sets conditions for the higher echelon to continue, or changes the higher "
            "echelon's scheme of maneuver. FM 3-0 (para 3-98 to 3-99) further reinforces that "
            "the main effort designation and resource priority decisions rest with the higher "
            "echelon commander, who must approve shifts that affect the overall operation. "
            "Therefore, bypassing is only permitted without approval if it falls within the "
            "commander's intent and does not change the higher echelon scheme of maneuver."
        ),
        section_references=[
            "FM 3-90, Chapter 1, para 1-78 to 1-82 (Actions on Contact - Choose an Action)",
            "FM 3-90, Chapter 1, para 1-80 (Conditions requiring higher approval)",
            "FM 3-0, Chapter 3, para 3-98 to 3-99 (Designate, Weight, and Sustain the Main Effort)",
        ],
        information_units=[
            "FM 3-90 lists bypass as one of five actions on contact (attack, bypass, defend, delay, withdrawal)",
            "FM 3-90 para 1-80 requires higher commander approval when the action requires additional resources",
            "Higher approval required when action is not within commander's intent",
            "Higher approval required when action changes the higher echelon's scheme of maneuver",
            "FM 3-0 states only the higher echelon commander designates and shifts the main effort",
            "The higher echelon commander always has the option to disapprove the unit in contact's action",
        ],
        source_documents=["FM 3-90", "FM 3-0"],
        category=QueryCategory.TRAP_A,
        hop_count=HopCount.TWO,
        difficulty="hard",
    ),
    GoldAnnotation(
        id="trap_a_02",
        query="In a mobile defense, what authority does the striking force commander have to initiate the counterattack, and are there any overriding conditions from the higher echelon?",
        ground_truth_answer=(
            "FM 3-90 (Chapter 10, para 10-6 to 10-9) describes the mobile defense where the "
            "striking force delivers the decisive blow. The striking force commander plans and "
            "prepares the counterattack. However, FM 3-90 para 10-6 states the striking force "
            "does not engage until the division or corps commander directs it. FM 6-0 (Chapter 1, "
            "para 1-5 to 1-6) establishes that commanders may delegate authority but delegation "
            "does not absolve commanders of responsibility to the higher echelon commander. "
            "FM 3-0 (Chapter 6, Section II) adds that the transition from defense to offense is "
            "a critical planning responsibility and commanders establish clear conditions and "
            "decision points for execution. Therefore, the striking force commander does NOT have "
            "independent authority to initiate—the higher echelon commander must direct the "
            "commitment based on pre-established decision points."
        ),
        section_references=[
            "FM 3-90, Chapter 10, para 10-1 to 10-9 (Mobile Defense)",
            "FM 6-0, Chapter 1, para 1-5 to 1-6 (Authority and Responsibility)",
            "FM 3-0, Chapter 6, Section II (Defensive Operations - Transition to Offense)",
        ],
        information_units=[
            "In a mobile defense, the striking force delivers the decisive blow via counterattack",
            "The striking force does not engage until directed by the division or corps commander",
            "FM 6-0 states delegation of authority does not absolve commanders of responsibility",
            "FM 3-0 requires clear conditions and decision points for defense-to-offense transitions",
            "The higher echelon commander retains commitment authority over the striking force",
        ],
        source_documents=["FM 3-90", "FM 6-0", "FM 3-0"],
        category=QueryCategory.TRAP_A,
        hop_count=HopCount.THREE,
        difficulty="hard",
    ),
    GoldAnnotation(
        id="trap_a_03",
        query="Can a unit in the defense independently withdraw from its battle position when it assesses it has been defeated?",
        ground_truth_answer=(
            "No. FM 3-90 Chapter 11 describes the retrograde, including withdrawal, and states "
            "that retrograde operations are typically conducted to preserve the force. However, "
            "FM 3-90 para 8-24 describes transitions from the defense and notes that the commander "
            "decides whether to transition. FM 6-0 (para 1-5 to 1-6) establishes that commanders "
            "are responsible for decisions and must operate within higher commander's intent. "
            "FM 3-0 (para 3-92 to 3-97) further specifies that transitions are critical planning "
            "responsibilities for commanders and that staffs monitor conditions that require "
            "transition. Commanders establish decision points to support transitions. A unilateral "
            "withdrawal without higher approval could change the scheme of maneuver and expose "
            "adjacent units. The exception is if the commander's intent or unit SOP explicitly "
            "provides authority for withdrawal under specified conditions (FM 3-90, para 1-82)."
        ),
        section_references=[
            "FM 3-90, Chapter 11 (Retrograde)",
            "FM 3-90, Chapter 8, para 8-24 (Defense Transitions)",
            "FM 6-0, Chapter 1, para 1-5 to 1-6 (Command Authority)",
            "FM 3-0, Chapter 3, para 3-92 to 3-97 (Transitions)",
        ],
        information_units=[
            "FM 3-90 describes withdrawal as a type of retrograde operation",
            "Transitions from defense are critical planning responsibilities per FM 3-0",
            "Commanders establish decision points for transitions during planning",
            "FM 6-0 requires actions within higher commander's intent",
            "Unilateral withdrawal could expose adjacent units and change the scheme of maneuver",
            "Exception: if commander's intent or SOP explicitly authorizes withdrawal under specified conditions",
        ],
        source_documents=["FM 3-90", "FM 6-0", "FM 3-0"],
        category=QueryCategory.TRAP_A,
        hop_count=HopCount.THREE,
        difficulty="hard",
    ),
    GoldAnnotation(
        id="trap_a_04",
        query="Does a division G-2 have the authority to task intelligence collection assets independently to answer a division priority intelligence requirement?",
        ground_truth_answer=(
            "The division G-2 manages intelligence collection for the division commander "
            "(FM 2-0, Chapter 7, Section VI). However, FM 2-0 Chapter 5 (para 5-8 to 5-10) "
            "describes collection management and states that the collection manager (G-2/S-2) "
            "develops and submits collection requirements. FM 6-0 (Chapter 2) establishes that "
            "staff members work within the commander's guidance and coordinate laterally. "
            "FM 2-0 Chapter 3 (para 3-15 to 3-18) states that information collection is "
            "a commander-driven activity integrated with operations. The operations staff (G-3) "
            "has tasking authority over maneuver units conducting reconnaissance. Therefore, "
            "while the G-2 manages intelligence requirements and recommends collection, actual "
            "tasking of maneuver reconnaissance assets requires the G-3's authority and the "
            "commander's approval through the information collection plan."
        ),
        section_references=[
            "FM 2-0, Chapter 7, Section VI (Division Intelligence)",
            "FM 2-0, Chapter 5, para 5-8 to 5-10 (Collection Management)",
            "FM 2-0, Chapter 3, para 3-15 to 3-18 (Information Collection)",
            "FM 6-0, Chapter 2 (Staff Roles and Responsibilities)",
        ],
        information_units=[
            "The G-2 manages intelligence collection for the commander",
            "Collection management involves developing and submitting requirements",
            "Information collection is a commander-driven activity integrated with operations",
            "The G-3 has tasking authority over maneuver units conducting reconnaissance",
            "The G-2 recommends collection; the commander approves through the information collection plan",
            "Staff coordination between G-2 and G-3 is required per FM 6-0",
        ],
        source_documents=["FM 2-0", "FM 6-0"],
        category=QueryCategory.TRAP_A,
        hop_count=HopCount.TWO,
        difficulty="hard",
    ),
    GoldAnnotation(
        id="trap_a_05",
        query="During large-scale combat operations, can the sustainment brigade commander independently reroute supply convoys to bypass a contested main supply route?",
        ground_truth_answer=(
            "FM 4-0 describes sustainment operations and the sustainment brigade's role. "
            "However, FM 3-0 (para 3-151 to 3-165) establishes that within assigned areas, "
            "movement control is a responsibility of the unit assigned the area of operations. "
            "FM 3-90 (para 1-53 to 1-54) states that terrain management requires coordination "
            "when moving through another unit's assigned area. FM 6-0 (Chapter 9) addresses "
            "command post operations and the importance of synchronizing movement. FM 4-0 "
            "Chapter 5 discusses the distribution network and states that movement control is "
            "coordinated through the movement control center. Therefore, the sustainment brigade "
            "commander cannot unilaterally reroute convoys through another unit's area of "
            "operations without coordinating with the owning unit and movement control authorities. "
            "The corps or division G-4 and movement control center coordinate route changes."
        ),
        section_references=[
            "FM 4-0, Chapter 5 (Sustainment During Armed Conflict)",
            "FM 3-0, Chapter 3, para 3-137 to 3-140 (Assigned Areas and AO Responsibilities)",
            "FM 3-90, Chapter 1, para 1-53 to 1-54 (Terrain Management)",
            "FM 6-0, Chapter 9 (Command Post Operations)",
        ],
        information_units=[
            "Movement control is a responsibility of the unit owning the area of operations",
            "FM 3-90 requires coordination when moving through another unit's assigned area",
            "The movement control center coordinates route changes",
            "The sustainment brigade cannot unilaterally reroute through another unit's AO",
            "Corps or division G-4 must coordinate route changes",
        ],
        source_documents=["FM 4-0", "FM 3-0", "FM 3-90", "FM 6-0"],
        category=QueryCategory.TRAP_A,
        hop_count=HopCount.THREE,
        difficulty="hard",
    ),
    GoldAnnotation(
        id="trap_a_06",
        query="Can a corps commander employ fires into a subordinate division's deep area without coordination?",
        ground_truth_answer=(
            "FM 3-0 (para 3-139, 3-151 to 3-165) describes the operational framework including "
            "deep, close, and rear operations. A zone assigned to a subordinate unit allows the "
            "higher headquarters to adjust deep operations without changing unit boundaries. "
            "However, FM 3-90 (Appendix A) and FM 3-0 describe fire support coordination measures "
            "including the coordinated fire line (CFL) and fire support coordination line (FSCL). "
            "FM 3-0 states that the higher headquarters uses fire support coordination and maneuver "
            "control measures to synchronize deep operations. When a subordinate is assigned a zone, "
            "the corps retains responsibility for synchronizing deep operations forward of the "
            "subordinate's coordinated fire line. If assigned an AO, the subordinate has fires "
            "clearance authority. Therefore, whether the corps can fire without coordination depends "
            "on the type of assigned area and the fire support coordination measures in effect."
        ),
        section_references=[
            "FM 3-0, Chapter 3, para 3-139 (Zones)",
            "FM 3-0, Chapter 3, para 3-140 (Sectors)",
            "FM 3-0, Chapter 3, para 3-151 to 3-165 (Deep, Close, and Rear Operations)",
            "FM 3-90, Appendix A (Tactical Control Measures)",
        ],
        information_units=[
            "In a zone, the higher HQ retains responsibility for deep operations synchronization",
            "In an AO, the subordinate unit has fires clearance authority within its boundaries",
            "Fire support coordination measures (CFL, FSCL) regulate fires authority",
            "The corps uses fire support coordination measures to synchronize with subordinate units",
            "The answer depends on whether the division has a zone, sector, or AO",
        ],
        source_documents=["FM 3-0", "FM 3-90"],
        category=QueryCategory.TRAP_A,
        hop_count=HopCount.TWO,
        difficulty="hard",
    ),
    GoldAnnotation(
        id="trap_a_07",
        query="During the MDMP, can the intelligence staff (G-2) independently change the priority intelligence requirements after the commander's planning guidance has been issued?",
        ground_truth_answer=(
            "FM 5-0 (Chapter 5) describes the MDMP steps. After the commander issues planning "
            "guidance (Step 2, Mission Analysis), the staff proceeds with course of action "
            "development. FM 2-0 (Chapter 3, para 3-2 to 3-4) states that the commander is the "
            "primary intelligence consumer and drives intelligence priorities. Priority intelligence "
            "requirements (PIRs) are the commander's requirements. FM 6-0 (Chapter 1, para 1-3) "
            "reinforces that commanders assess situations, make decisions, and direct action—the "
            "staff supports the commander's decision making. FM 2-0 states the G-2 recommends PIRs "
            "but the commander approves them. Therefore, the G-2 cannot independently change PIRs "
            "after the commander's guidance; the G-2 must recommend changes to the commander for "
            "approval. Only the commander can modify PIRs."
        ),
        section_references=[
            "FM 5-0, Chapter 5 (MDMP - Step 2 Mission Analysis)",
            "FM 2-0, Chapter 3, para 3-2 to 3-4 (Role of the Commander in Intelligence)",
            "FM 6-0, Chapter 1, para 1-3 (Commanders as Focal Point of C2)",
        ],
        information_units=[
            "PIRs are the commander's requirements, not the staff's",
            "The G-2 recommends PIRs but the commander approves them",
            "After commander's planning guidance, staff works within that guidance",
            "FM 6-0 states the commander makes decisions and directs action",
            "Only the commander can modify PIRs",
        ],
        source_documents=["FM 5-0", "FM 2-0", "FM 6-0"],
        category=QueryCategory.TRAP_A,
        hop_count=HopCount.THREE,
        difficulty="hard",
    ),
    GoldAnnotation(
        id="trap_a_08",
        query="Can a BCT conduct a passage of lines through an adjacent unit's sector without the division commander's approval?",
        ground_truth_answer=(
            "No. FM 3-90 (Chapter 16, para 16-1 to 16-8) describes the passage of lines as "
            "an operation requiring detailed coordination between the passing and stationary forces. "
            "FM 3-90 states that the higher headquarters (division) directs a passage of lines and "
            "specifies the passing unit, stationary unit, and passage points. FM 3-0 (para 3-137 "
            "to 3-140) establishes that units must coordinate when operating in another unit's "
            "assigned area. A BCT passing through another BCT's sector requires the division "
            "commander to direct the passage, establish passage points, coordinate the battle "
            "handover line, and synchronize fires and maneuver between the two BCTs."
        ),
        section_references=[
            "FM 3-90, Chapter 16, para 16-1 to 16-8 (Passage of Lines)",
            "FM 3-0, Chapter 3, para 3-137 to 3-140 (Assigned Areas)",
        ],
        information_units=[
            "A passage of lines is directed by the higher headquarters",
            "The higher HQ specifies passing unit, stationary unit, and passage points",
            "Coordination between passing and stationary forces is required",
            "The division establishes the battle handover line",
            "Movement through another unit's assigned area requires coordination per FM 3-0",
        ],
        source_documents=["FM 3-90", "FM 3-0"],
        category=QueryCategory.TRAP_A,
        hop_count=HopCount.TWO,
        difficulty="medium",
    ),
    GoldAnnotation(
        id="trap_a_09",
        query="During armed conflict, can the theater sustainment command establish supply points in the corps rear area independently?",
        ground_truth_answer=(
            "FM 4-0 (Chapter 2, Section III) describes the theater sustainment command's role, "
            "including establishing distribution networks. However, FM 3-0 (para 3-151 to 3-165) "
            "and FM 4-0 (Chapter 5, Section I) establish that sustainment activities within the "
            "corps area of operations must be coordinated with the corps headquarters. FM 4-0 "
            "discusses that rear operations include security and terrain management. FM 3-90 "
            "(para 1-53 to 1-54) requires coordination when establishing activities within another "
            "unit's assigned area. The corps G-4 coordinates sustainment within the corps AO. "
            "Therefore, the theater sustainment command must coordinate with the corps to establish "
            "supply points in the corps rear area, as the corps commander is responsible for "
            "terrain management and security within the corps AO."
        ),
        section_references=[
            "FM 4-0, Chapter 2, Section III (Theater Strategic Level)",
            "FM 4-0, Chapter 5, Section I (Sustainment During Armed Conflict)",
            "FM 3-0, Chapter 3, para 3-151 to 3-165 (Deep, Close, and Rear Operations)",
            "FM 3-90, Chapter 1, para 1-53 to 1-54 (Terrain Management)",
        ],
        information_units=[
            "Theater sustainment command establishes distribution networks",
            "Activities in the corps AO require coordination with the corps HQ",
            "The corps commander is responsible for terrain management in the corps AO",
            "The corps G-4 coordinates sustainment within the corps AO",
            "Rear operations include security of sustainment nodes",
        ],
        source_documents=["FM 4-0", "FM 3-0", "FM 3-90"],
        category=QueryCategory.TRAP_A,
        hop_count=HopCount.THREE,
        difficulty="hard",
    ),
    GoldAnnotation(
        id="trap_a_10",
        query="Can a division commander re-task a brigade's organic military intelligence company for a division-level collection mission?",
        ground_truth_answer=(
            "FM 2-0 (Chapter 7, Section VII) describes BCT intelligence capabilities including "
            "the organic military intelligence company. FM 2-0 states these assets provide the BCT "
            "commander with organic collection capability. However, FM 2-0 (Chapter 6, para 6-13 "
            "to 6-16) describes task-organizing intelligence assets and states that the higher "
            "headquarters may re-task or cross-attach intelligence assets based on the situation. "
            "FM 3-0 (Appendix B) and FM 5-0 (Appendix B) describe Army command relationships "
            "and establish that the division commander has the authority to task-organize subordinate "
            "BCT assets when the situation requires it. However, this should be done judiciously, "
            "as it removes capability from the BCT. The division commander has the authority but "
            "should weigh the impact on the BCT's ability to accomplish its mission."
        ),
        section_references=[
            "FM 2-0, Chapter 7, Section VII (BCT Intelligence)",
            "FM 2-0, Chapter 6, para 6-13 to 6-16 (Task-Organizing Intelligence)",
            "FM 3-0, Appendix B (Command and Support Relationships)",
            "FM 5-0, Appendix B (Command and Support Relationships)",
        ],
        information_units=[
            "The military intelligence company is organic to the BCT",
            "Higher HQ may re-task or cross-attach intelligence assets",
            "The division commander has task-organization authority over subordinate BCT assets",
            "Re-tasking removes capability from the BCT",
            "Command relationships determine task-organization authority",
            "Division commander should weigh impact on BCT mission accomplishment",
        ],
        source_documents=["FM 2-0", "FM 3-0", "FM 5-0"],
        category=QueryCategory.TRAP_A,
        hop_count=HopCount.THREE,
        difficulty="hard",
    ),

    # =========================================================================
    # TRAP B: "The Distant Definition" (10 queries)
    # Query uses a doctrinal term defined in a different FM or chapter
    # =========================================================================
    GoldAnnotation(
        id="trap_b_01",
        query="How does convergence apply to intelligence operations during large-scale combat operations?",
        ground_truth_answer=(
            "Convergence is defined in FM 3-0 (Chapter 3, para 3-48 to 3-54) as the integration "
            "of capabilities from multiple domains, the EMS, and the information environment to "
            "create effects that achieve objectives. FM 3-0 states that convergence requires a "
            "tempo and timing of capabilities to create windows of opportunity. FM 2-0 (Chapter 2, "
            "para 2-21 to 2-26) describes how intelligence drives multidomain operations by "
            "finding windows of opportunity through building situational understanding. Intelligence "
            "enables convergence by providing the common intelligence picture that allows commanders "
            "to synchronize capabilities from multiple domains against enemy vulnerabilities at "
            "decisive points. FM 2-0 (Chapter 8) describes how intelligence supports identifying "
            "targets for the integration of fires from multiple domains."
        ),
        section_references=[
            "FM 3-0, Chapter 3, para 3-48 to 3-54 (Convergence - definition and concept)",
            "FM 2-0, Chapter 2, para 2-21 to 2-26 (Intelligence and Multidomain Operations)",
            "FM 2-0, Chapter 8 (Intelligence During LSCO)",
        ],
        information_units=[
            "Convergence is defined in FM 3-0 as integration of capabilities from multiple domains",
            "Convergence requires timing and tempo to create windows of opportunity",
            "Intelligence drives multidomain operations per FM 2-0",
            "Intelligence enables convergence by providing the common intelligence picture",
            "Intelligence supports identifying targets for multi-domain fires integration",
        ],
        source_documents=["FM 3-0", "FM 2-0"],
        category=QueryCategory.TRAP_B,
        hop_count=HopCount.TWO,
        difficulty="hard",
    ),
    GoldAnnotation(
        id="trap_b_02",
        query="What is the role of the 'decisive point' in planning an attack at the brigade level?",
        ground_truth_answer=(
            "A decisive point is defined in FM 3-0 (para 3-112) as key terrain, key event, "
            "critical factor, or function that, when acted upon, enables commanders to gain a "
            "marked advantage over an enemy or contribute materially to achieving success (JP 5-0). "
            "FM 5-0 (Chapter 2, para 38 to 45) discusses decisive points as an element of "
            "operational art used during planning. In planning an attack, FM 3-90 (Chapter 5, "
            "para 5-3 to 5-5) describes how commanders identify the enemy's vulnerabilities and "
            "plan to concentrate combat power at the decisive point. The decisive point is where "
            "the commander plans to achieve the purpose of the attack by finishing the enemy. "
            "FM 3-0 (para 3-113) notes that peer threats require attacking combinations of "
            "decisive points across multiple domains."
        ),
        section_references=[
            "FM 3-0, Chapter 3, para 3-112 to 3-113 (Decisive Points definition)",
            "FM 5-0, Chapter 2 (Planning and Operational Art - elements of operational art)",
            "FM 3-90, Chapter 5, para 5-3 to 5-5 (Planning for an Attack)",
        ],
        information_units=[
            "Decisive point defined as key terrain/event/factor enabling marked advantage (FM 3-0/JP 5-0)",
            "Decisive points are an element of operational art used in planning (FM 5-0)",
            "In an attack, commanders concentrate combat power at the decisive point (FM 3-90)",
            "Peer threats require attacking combinations of decisive points across domains (FM 3-0)",
            "The decisive point is where the commander plans to finish the enemy",
        ],
        source_documents=["FM 3-0", "FM 5-0", "FM 3-90"],
        category=QueryCategory.TRAP_B,
        hop_count=HopCount.THREE,
        difficulty="hard",
    ),
    GoldAnnotation(
        id="trap_b_03",
        query="How does the concept of 'operational reach' affect sustainment planning during offensive operations?",
        ground_truth_answer=(
            "Operational reach is defined in FM 3-0 as the distance and duration across which "
            "a force can successfully employ military capabilities (ADP 3-0). FM 4-0 (Chapter 1, "
            "Section IV) states that sustainment enables freedom of action, extends operational "
            "reach, and prolongs endurance. FM 4-0 (Chapter 5, Section III) describes sustainment "
            "of offensive operations and states that sustainment determines the depth and duration "
            "of Army operations. As units advance, their lines of communication lengthen, increasing "
            "vulnerability and reducing the speed of resupply. FM 4-0 discusses how commanders must "
            "position supplies forward and use logistics release points to maintain operational reach. "
            "FM 3-0 (para 3-98) adds that commanders balance forward positioning of sustainment "
            "assets with freedom of action when weighting the main effort."
        ),
        section_references=[
            "FM 3-0, Chapter 3 (Operational Reach definition - from ADP 3-0)",
            "FM 4-0, Chapter 1, Section IV (Sustainment Support to MDO)",
            "FM 4-0, Chapter 5, Section III (Sustainment of Offensive Operations)",
            "FM 3-0, Chapter 3, para 3-98 (Weighting the Main Effort)",
        ],
        information_units=[
            "Operational reach is the distance and duration a force can employ capabilities",
            "Sustainment extends operational reach and prolongs endurance (FM 4-0)",
            "Sustainment determines the depth and duration of operations",
            "Lengthening lines of communication increase vulnerability during offense",
            "Commanders position supplies forward using logistics release points",
            "Commanders balance forward sustainment positioning with freedom of action (FM 3-0)",
        ],
        source_documents=["FM 3-0", "FM 4-0"],
        category=QueryCategory.TRAP_B,
        hop_count=HopCount.TWO,
        difficulty="medium",
    ),
    GoldAnnotation(
        id="trap_b_04",
        query="What is 'mission command' and how does it affect the way intelligence operations are conducted at the battalion level?",
        ground_truth_answer=(
            "Mission command is defined in FM 6-0 (para 1-15 to 1-18) as the Army's approach "
            "to command and control that empowers subordinate decision making and decentralized "
            "execution appropriate to the situation (ADP 6-0). It is enabled by competence, mutual "
            "trust, shared understanding, commander's intent, mission orders, disciplined initiative, "
            "and risk acceptance. FM 2-0 (Chapter 7, Section VIII) describes battalion intelligence "
            "and states the battalion S-2 must be prepared to operate with degraded communications "
            "and limited higher echelon support. Mission command principles mean the battalion S-2 "
            "must exercise disciplined initiative to continue intelligence operations within the "
            "commander's intent, even when they cannot reach higher echelons for guidance. "
            "FM 2-0 (Chapter 1, para 1-8 to 1-13) describes how the intelligence process must "
            "support decentralized execution."
        ),
        section_references=[
            "FM 6-0, Chapter 1, para 1-15 to 1-18 (Mission Command definition and principles)",
            "FM 2-0, Chapter 7, Section VIII (Battalion Intelligence)",
            "FM 2-0, Chapter 1, para 1-8 to 1-13 (Intelligence Process)",
        ],
        information_units=[
            "Mission command empowers subordinate decision making and decentralized execution",
            "Mission command is enabled by seven principles including disciplined initiative",
            "Battalion S-2 must operate with potentially degraded communications",
            "Battalion S-2 exercises disciplined initiative within commander's intent",
            "Intelligence operations continue even without higher echelon guidance",
            "Intelligence process must support decentralized execution",
        ],
        source_documents=["FM 6-0", "FM 2-0"],
        category=QueryCategory.TRAP_B,
        hop_count=HopCount.TWO,
        difficulty="medium",
    ),
    GoldAnnotation(
        id="trap_b_05",
        query="How does the 'common operational picture' support the commander during the execution of defensive operations?",
        ground_truth_answer=(
            "The common operational picture (COP) is described in FM 6-0 (Chapter 6, para 6-7 "
            "to 6-8) as a display of relevant information within a commander's area of interest "
            "tailored to the user's requirements. FM 6-0 Table 6-1 provides an example COP "
            "checklist including friendly force locations, enemy situation, and logistics status. "
            "FM 3-90 (Chapter 8) describes defensive operations and the importance of maintaining "
            "situational awareness of enemy forces and friendly positions. FM 2-0 (Chapter 5, "
            "para 5-24 to 5-26) describes the common intelligence picture as the intelligence "
            "portion of the COP. During defensive operations, the COP enables the commander to "
            "track enemy penetrations, assess the status of the defense, identify when to commit "
            "the reserve, and determine when transition conditions are met. FM 5-0 (Chapter 6) "
            "describes how the COP supports the rapid decision-making and synchronization process "
            "during execution."
        ),
        section_references=[
            "FM 6-0, Chapter 6, para 6-7 to 6-8 (Common Operational Picture)",
            "FM 3-90, Chapter 8 (The Defense)",
            "FM 2-0, Chapter 5, para 5-24 to 5-26 (Common Intelligence Picture)",
            "FM 5-0, Chapter 6 (Decision Making During Execution - RDSP)",
        ],
        information_units=[
            "COP is a display of relevant information within the commander's area of interest",
            "COP includes friendly force locations, enemy situation, and logistics status",
            "Common intelligence picture is the intelligence portion of the COP (FM 2-0)",
            "In defense, COP helps track enemy penetrations and assess defense status",
            "COP supports deciding when to commit the reserve",
            "COP supports the RDSP during execution (FM 5-0)",
        ],
        source_documents=["FM 6-0", "FM 3-90", "FM 2-0", "FM 5-0"],
        category=QueryCategory.TRAP_B,
        hop_count=HopCount.FOUR,
        difficulty="hard",
    ),
    GoldAnnotation(
        id="trap_b_06",
        query="What is the 'operations process' and how does it integrate intelligence preparation of the operational environment?",
        ground_truth_answer=(
            "The operations process is described in FM 6-0 (Chapter 1, para 1-7, Figure 1-3) "
            "as the major activities of planning, preparing, executing, and continuously assessing. "
            "FM 5-0 (Chapter 1, Figure 1-1) further describes the operations process and how "
            "planning drives the other activities. Intelligence preparation of the operational "
            "environment (IPOE) is described in FM 2-0 (Chapter 5, para 5-14 to 5-18) and in "
            "FM 5-0 (Appendix G, Figure G-2) which shows how IPOE integrates into the MDMP. "
            "IPOE defines the operational environment, describes environmental effects, evaluates "
            "the threat, and determines threat courses of action. IPOE feeds directly into mission "
            "analysis (Step 2 of MDMP) and continues throughout execution as situation development."
        ),
        section_references=[
            "FM 6-0, Chapter 1, para 1-7, Figure 1-3 (Operations Process)",
            "FM 5-0, Chapter 1, Figure 1-1 (Operations Process and Planning)",
            "FM 2-0, Chapter 5, para 5-14 to 5-18 (IPOE)",
            "FM 5-0, Appendix G, Figure G-2 (IPOE within MDMP)",
        ],
        information_units=[
            "Operations process consists of planning, preparing, executing, and assessing",
            "FM 6-0 and FM 5-0 both describe the operations process",
            "IPOE defines the OE, describes environmental effects, evaluates threat, determines threat COA",
            "IPOE integrates into mission analysis (Step 2 of MDMP)",
            "IPOE continues throughout execution as situation development",
        ],
        source_documents=["FM 6-0", "FM 5-0", "FM 2-0"],
        category=QueryCategory.TRAP_B,
        hop_count=HopCount.THREE,
        difficulty="medium",
    ),
    GoldAnnotation(
        id="trap_b_07",
        query="What are 'defeat mechanisms' and how does the sustainment warfighting function contribute to the 'isolate' mechanism?",
        ground_truth_answer=(
            "Defeat mechanisms are defined in FM 3-0 (para 3-114 to 3-121) as methods through "
            "which friendly forces accomplish their mission against enemy opposition. The four "
            "defeat mechanisms are destroy, dislocate, disintegrate, and isolate. Isolate means "
            "to separate a force from its sources of support to reduce its effectiveness and "
            "increase its vulnerability to defeat (FM 3-0, para 3-119). FM 4-0 describes how "
            "sustainment contributes to operations and FM 3-0 (para 3-121) states that destroying "
            "enemy sustainment capability separates enemy fires and maneuver from fuel and "
            "ammunition and delays resupply operations. This directly contributes to the isolate "
            "mechanism. FM 4-0 (Chapter 5) describes how targeting enemy logistics nodes and "
            "lines of communication through interdiction contributes to isolating enemy formations."
        ),
        section_references=[
            "FM 3-0, Chapter 3, para 3-114 to 3-121 (Defeat Mechanisms)",
            "FM 3-0, Chapter 3, para 3-119 (Isolate definition)",
            "FM 3-0, Chapter 3, para 3-121 (Destroying enemy sustainment)",
            "FM 4-0, Chapter 5 (Sustainment During Armed Conflict)",
        ],
        information_units=[
            "Four defeat mechanisms: destroy, dislocate, disintegrate, isolate",
            "Isolate means separating a force from its sources of support",
            "Destroying enemy sustainment separates enemy from fuel and ammunition",
            "Targeting enemy logistics contributes to the isolate mechanism",
            "Interdiction of lines of communication isolates enemy formations",
        ],
        source_documents=["FM 3-0", "FM 4-0"],
        category=QueryCategory.TRAP_B,
        hop_count=HopCount.TWO,
        difficulty="medium",
    ),
    GoldAnnotation(
        id="trap_b_08",
        query="What is 'relative advantage' and how does the intelligence warfighting function help create it?",
        ground_truth_answer=(
            "A relative advantage is defined in FM 3-0 (para 1-12) as a location or condition, "
            "in any domain, relative to an adversary or enemy that provides an opportunity to "
            "progress towards or achieve an objective (ADP 3-0). Commanders seek and create "
            "relative advantages to exploit through action. FM 2-0 (Chapter 2, para 2-21 to 2-24) "
            "describes how intelligence enables finding windows of opportunity by building "
            "situational understanding. FM 2-0 states that intelligence drives multidomain "
            "operations by identifying enemy vulnerabilities and capabilities, enabling commanders "
            "to create and exploit relative advantages. FM 2-0 Figure 2-7 shows how building "
            "situational understanding identifies windows of opportunity that represent relative "
            "advantages."
        ),
        section_references=[
            "FM 3-0, Chapter 1, para 1-12 (Relative Advantage definition)",
            "FM 2-0, Chapter 2, para 2-21 to 2-24 (Intelligence and windows of opportunity)",
            "FM 2-0, Figure 2-7 (Finding windows of opportunity)",
        ],
        information_units=[
            "Relative advantage is a location or condition providing opportunity to achieve objectives",
            "Commanders seek and create relative advantages to exploit",
            "Intelligence builds situational understanding to find windows of opportunity",
            "Intelligence identifies enemy vulnerabilities and capabilities",
            "Windows of opportunity represent relative advantages",
        ],
        source_documents=["FM 3-0", "FM 2-0"],
        category=QueryCategory.TRAP_B,
        hop_count=HopCount.TWO,
        difficulty="medium",
    ),
    GoldAnnotation(
        id="trap_b_09",
        query="What is 'systems warfare' and how should intelligence preparation of the operational environment account for it?",
        ground_truth_answer=(
            "Systems warfare is described in FM 3-0 (para 2-45 to 2-47) as the identification "
            "and isolation or destruction of critical subsystems or components to degrade or "
            "destroy an opponent's overall system. Peer threats view the battlefield as a collection "
            "of complex, dynamic, and integrated systems. FM 2-0 (Chapter 5, para 5-14 to 5-18) "
            "describes IPOE as the systematic process of analyzing the operational environment. "
            "FM 2-0 (Chapter 2, Table 2-1) describes intelligence considerations for Army strategic "
            "challenges. To account for systems warfare, IPOE must identify the enemy's critical "
            "subsystems and the dependencies between them, map the interactions and linkages within "
            "the enemy's combat system, and identify which components are most vulnerable to "
            "disruption. FM 2-0 (Chapter 8, para 8-23 to 8-32) describes intelligence support to "
            "targeting, which directly supports systems warfare by identifying high-payoff targets "
            "within enemy systems."
        ),
        section_references=[
            "FM 3-0, Chapter 2, para 2-45 to 2-47 (Systems Warfare)",
            "FM 2-0, Chapter 5, para 5-14 to 5-18 (IPOE)",
            "FM 2-0, Chapter 2, Table 2-1 (Intelligence Considerations)",
            "FM 2-0, Chapter 8, para 8-23 to 8-32 (Intelligence Support to Targeting)",
        ],
        information_units=[
            "Systems warfare targets critical subsystems to degrade the overall system",
            "Peer threats view the battlefield as integrated systems",
            "IPOE must identify enemy critical subsystems and dependencies",
            "IPOE maps interactions and linkages within the enemy combat system",
            "Intelligence support to targeting identifies high-payoff targets in enemy systems",
        ],
        source_documents=["FM 3-0", "FM 2-0"],
        category=QueryCategory.TRAP_B,
        hop_count=HopCount.TWO,
        difficulty="hard",
    ),
    GoldAnnotation(
        id="trap_b_10",
        query="What is a 'running estimate' and how does the sustainment staff use it during large-scale combat operations?",
        ground_truth_answer=(
            "A running estimate is described in FM 5-0 (Appendix C) as a staff product that "
            "tracks the current situation and projects future requirements to support the "
            "commander's decision making. FM 5-0 Table C-1 provides a generic running estimate "
            "format. FM 6-0 (Chapter 2, para 2-2) describes that a common staff duty is to "
            "maintain running estimates within their areas of expertise. FM 4-0 (Chapter 5) "
            "describes how the sustainment staff maintains a running estimate of supply status, "
            "maintenance capabilities, transportation assets, and health service support. During "
            "LSCO, the sustainment running estimate feeds into the commander's decision making "
            "for committing reserves, shifting the main effort, and determining culmination points."
        ),
        section_references=[
            "FM 5-0, Appendix C (Running Estimates)",
            "FM 6-0, Chapter 2, para 2-2 (Common Staff Duties - running estimates)",
            "FM 4-0, Chapter 5 (Sustainment During Armed Conflict)",
        ],
        information_units=[
            "A running estimate tracks current situation and projects future requirements",
            "FM 5-0 Appendix C provides the generic format",
            "Staff members maintain running estimates within their areas per FM 6-0",
            "Sustainment running estimate tracks supply, maintenance, transport, health service",
            "Running estimate supports commander's decisions on reserves and main effort shifts",
        ],
        source_documents=["FM 5-0", "FM 6-0", "FM 4-0"],
        category=QueryCategory.TRAP_B,
        hop_count=HopCount.THREE,
        difficulty="medium",
    ),

    # =========================================================================
    # TRAP C: "The Scattered Components" (10 queries)
    # Answer requires collecting components from multiple sections or FMs
    # =========================================================================
    GoldAnnotation(
        id="trap_c_01",
        query="What are all the warfighting functions and how does each one contribute to large-scale combat operations?",
        ground_truth_answer=(
            "The six warfighting functions are described across FM 3-0 (Chapter 2, para 2-1 to "
            "2-7) and expanded in their respective FMs. They are: (1) Command and Control (FM 6-0) "
            "- synchronizes and converges all elements of combat power; (2) Intelligence (FM 2-0) "
            "- provides understanding of the enemy, terrain, and weather; (3) Fires - provides "
            "collective and coordinated use of lethal and nonlethal fires; (4) Movement and "
            "Maneuver (FM 3-90) - moves and employs forces for position of advantage; "
            "(5) Protection - preserves the force so it can apply maximum combat power; "
            "(6) Sustainment (FM 4-0) - provides freedom of action, extends operational reach, "
            "and prolongs endurance. Each warfighting function contributes tasks and systems "
            "that commanders synchronize to generate combat power."
        ),
        section_references=[
            "FM 3-0, Chapter 2, para 2-1 to 2-7 (Warfighting Functions overview)",
            "FM 6-0, Chapter 1 (C2 Warfighting Function)",
            "FM 2-0, Chapter 1 (Intelligence Warfighting Function)",
            "FM 3-90, Chapter 1, para 1-90 to 1-91 (Movement and Maneuver WfF)",
            "FM 4-0, Chapter 1 (Sustainment Warfighting Function)",
        ],
        information_units=[
            "Command and Control synchronizes combat power (FM 6-0)",
            "Intelligence provides understanding of enemy, terrain, weather (FM 2-0)",
            "Fires provides coordinated lethal and nonlethal fires",
            "Movement and Maneuver moves forces for position of advantage (FM 3-90)",
            "Protection preserves the force",
            "Sustainment extends operational reach and prolongs endurance (FM 4-0)",
            "Warfighting functions are unified by a common purpose per FM 3-0",
        ],
        source_documents=["FM 3-0", "FM 6-0", "FM 2-0", "FM 3-90", "FM 4-0"],
        category=QueryCategory.TRAP_C,
        hop_count=HopCount.FOUR,
        difficulty="medium",
    ),
    GoldAnnotation(
        id="trap_c_02",
        query="List all the steps of the Military Decision-Making Process and describe the intelligence staff's role in each step.",
        ground_truth_answer=(
            "The MDMP has seven steps described in FM 5-0 Chapter 5: (1) Receipt of Mission, "
            "(2) Mission Analysis, (3) COA Development, (4) COA Analysis (War Game), "
            "(5) COA Comparison, (6) COA Approval, (7) Orders Production. FM 2-0 (Chapter 3, "
            "Table 3-3) describes intelligence support to each MDMP step. FM 5-0 (Appendix G, "
            "Figure G-1) shows the relationship between MDMP, IPOE, targeting, and information "
            "collection. Key G-2 roles include: conducting IPOE during mission analysis, developing "
            "threat COAs for war gaming, providing intelligence running estimates, supporting "
            "targeting during COA analysis, and producing Annex B (Intelligence) during orders "
            "production."
        ),
        section_references=[
            "FM 5-0, Chapter 5 (MDMP Steps 1-7)",
            "FM 2-0, Chapter 3, Table 3-3 (Intelligence Support to MDMP)",
            "FM 5-0, Appendix G (Integrating Processes - MDMP/IPOE/Targeting)",
            "FM 5-0, Appendix E (Annex B Intelligence format)",
        ],
        information_units=[
            "Step 1: Receipt of Mission - G-2 begins updating intelligence estimates",
            "Step 2: Mission Analysis - G-2 conducts IPOE and recommends PIRs",
            "Step 3: COA Development - G-2 develops threat COAs",
            "Step 4: COA Analysis - G-2 role-plays threat during war game",
            "Step 5: COA Comparison - G-2 provides intelligence assessment of each COA",
            "Step 6: COA Approval - G-2 refines intelligence estimate",
            "Step 7: Orders Production - G-2 produces Annex B (Intelligence)",
            "IPOE integrates throughout MDMP per FM 5-0 Appendix G",
        ],
        source_documents=["FM 5-0", "FM 2-0"],
        category=QueryCategory.TRAP_C,
        hop_count=HopCount.THREE,
        difficulty="hard",
    ),
    GoldAnnotation(
        id="trap_c_03",
        query="What are all the considerations for transitioning from defensive to offensive operations?",
        ground_truth_answer=(
            "Transition from defense to offense is discussed across multiple FMs: "
            "FM 3-0 (Chapter 6, Section II, para on Transition to Offense) describes "
            "establishing conditions for the transition. FM 3-90 (Chapter 8, para 8-24) describes "
            "tactical transitions from the defense. FM 5-0 describes planning transitions as "
            "branches and sequels. FM 2-0 (Chapter 8) describes intelligence requirements for "
            "identifying when the enemy has culminated. FM 4-0 (Chapter 5, Section II) describes "
            "sustainment considerations including pre-positioning supplies for the transition. "
            "FM 6-0 (Appendix C) describes rehearsals for transitions. Key considerations include: "
            "identifying enemy culmination, pre-positioning reserves, shifting sustainment priority "
            "to the attacking force, rehearsing transition actions, issuing fragmentary orders, "
            "and maintaining continuous intelligence on enemy dispositions."
        ),
        section_references=[
            "FM 3-0, Chapter 6, Section II (Transition to Offense)",
            "FM 3-90, Chapter 8, para 8-24 (Defensive Transitions)",
            "FM 5-0, Chapter 1 (Planning branches and sequels)",
            "FM 2-0, Chapter 8 (Intelligence for identifying enemy culmination)",
            "FM 4-0, Chapter 5, Section II (Sustainment of Defensive Operations - Transition)",
            "FM 6-0, Appendix C (Rehearsals for transitions)",
        ],
        information_units=[
            "Identify when the enemy has culminated (FM 2-0, FM 3-0)",
            "Pre-position reserves for counterattack (FM 3-90, FM 3-0)",
            "Shift sustainment priority to the attacking force (FM 4-0)",
            "Plan transitions as branches and sequels (FM 5-0)",
            "Rehearse transition actions (FM 6-0)",
            "Issue fragmentary orders for transition",
            "Maintain continuous intelligence on enemy dispositions (FM 2-0)",
            "Establish decision points for transition execution (FM 3-0)",
        ],
        source_documents=["FM 3-0", "FM 3-90", "FM 5-0", "FM 2-0", "FM 4-0", "FM 6-0"],
        category=QueryCategory.TRAP_C,
        hop_count=HopCount.FOUR,
        difficulty="hard",
    ),
    GoldAnnotation(
        id="trap_c_04",
        query="What are all the types of defensive operations and their distinguishing characteristics?",
        ground_truth_answer=(
            "FM 3-90 (Chapter 8, para 8-2 to 8-3) identifies the types of defensive operations: "
            "area defense, mobile defense, and retrograde. FM 3-90 Chapter 9 describes the area "
            "defense as concentrating on retaining terrain by absorbing the enemy into an "
            "interlocking series of positions. Variations include defense in depth, forward defense, "
            "perimeter defense, and defense of a linear obstacle. FM 3-90 Chapter 10 describes "
            "the mobile defense as using a combination of a fixing force and a striking force to "
            "defeat the enemy through a decisive counterattack. FM 3-90 Chapter 11 describes "
            "retrograde operations including delay, withdrawal, and retirement. FM 3-0 "
            "(Chapter 6, Section II) provides operational-level context for defensive operations "
            "and describes how the operational framework applies."
        ),
        section_references=[
            "FM 3-90, Chapter 8, para 8-2 to 8-3 (Types of Defensive Operations)",
            "FM 3-90, Chapter 9 (Area Defense and variations)",
            "FM 3-90, Chapter 10 (Mobile Defense)",
            "FM 3-90, Chapter 11 (Retrograde - Delay, Withdrawal, Retirement)",
            "FM 3-0, Chapter 6, Section II (Defensive Operations)",
        ],
        information_units=[
            "Area defense concentrates on retaining terrain with interlocking positions",
            "Area defense variations: defense in depth, forward defense, perimeter, linear obstacle",
            "Mobile defense uses fixing force and striking force with decisive counterattack",
            "Retrograde includes delay, withdrawal, and retirement",
            "Delay trades space for time",
            "Withdrawal moves the force away from the enemy",
            "Retirement is movement away from the enemy when not in contact",
            "FM 3-0 provides operational-level defensive framework context",
        ],
        source_documents=["FM 3-90", "FM 3-0"],
        category=QueryCategory.TRAP_C,
        hop_count=HopCount.TWO,
        difficulty="medium",
    ),
    GoldAnnotation(
        id="trap_c_05",
        query="What are all the principles of sustainment and how do they apply to large-scale combat operations?",
        ground_truth_answer=(
            "FM 4-0 (Chapter 1, Section I, Figure 1-1) identifies the principles of sustainment: "
            "integration, anticipation, responsiveness, simplicity, economy, survivability, "
            "continuity, and improvisation. Integration means linking sustainment to operations. "
            "Anticipation means predicting requirements. Responsiveness means providing the right "
            "support at the right time and place. Simplicity means avoiding unnecessary complexity. "
            "Economy means efficient use of resources. Survivability means protecting sustainment "
            "assets. Continuity means uninterrupted support. Improvisation means adapting to "
            "unexpected situations. During LSCO, FM 4-0 (Chapter 5) describes how survivability "
            "becomes critical as sustainment nodes are targeted by enemy fires, and anticipation "
            "is essential because consumption rates increase dramatically."
        ),
        section_references=[
            "FM 4-0, Chapter 1, Section I (Principles of Sustainment, Figure 1-1)",
            "FM 4-0, Chapter 5 (Sustainment During Armed Conflict)",
        ],
        information_units=[
            "Integration: linking sustainment to operations",
            "Anticipation: predicting requirements before they arise",
            "Responsiveness: right support at right time and place",
            "Simplicity: avoiding unnecessary complexity",
            "Economy: efficient use of resources",
            "Survivability: protecting sustainment assets from enemy fires",
            "Continuity: uninterrupted support",
            "Improvisation: adapting to unexpected situations",
            "During LSCO, survivability is critical as nodes are targeted",
            "Anticipation essential due to dramatically increased consumption rates",
        ],
        source_documents=["FM 4-0"],
        category=QueryCategory.TRAP_C,
        hop_count=HopCount.TWO,
        difficulty="medium",
    ),
    GoldAnnotation(
        id="trap_c_06",
        query="What are all the components of a command and control system and how are they organized in a command post?",
        ground_truth_answer=(
            "FM 6-0 (Chapter 1, para 1-22 to 1-24, Figure 1-2) identifies the four components "
            "of a C2 system: people, processes, networks, and command posts. People (Chapter 1, "
            "para 1-25 to 1-28) include commanders, staff, and liaisons. Processes (Chapter 4) "
            "include the battle rhythm, meetings, boards, and working groups. Networks (Chapter 6) "
            "include communications systems and the COP. Command posts (Chapters 7-9) include "
            "types (main, tactical, contingency), organization into functional and integrating "
            "cells (Chapter 8, Figure 8-1), and operations including shift schedules and SOPs "
            "(Chapter 9). FM 6-0 (Chapter 8) describes how the staff is organized into cells "
            "including the current operations cell, future operations cell, and plans cell."
        ),
        section_references=[
            "FM 6-0, Chapter 1, para 1-22 to 1-24 (C2 System Components)",
            "FM 6-0, Chapter 1, para 1-25 to 1-28 (People)",
            "FM 6-0, Chapter 4 (Battle Rhythm and Meetings - Processes)",
            "FM 6-0, Chapter 6 (Networks and Communications)",
            "FM 6-0, Chapters 7-9 (Command Posts)",
            "FM 6-0, Chapter 8, Figure 8-1 (Functional and Integrating Cells)",
        ],
        information_units=[
            "Four C2 system components: people, processes, networks, command posts",
            "People: commanders, staff, liaison officers",
            "Processes: battle rhythm, meetings, boards, working groups",
            "Networks: communications systems and COP",
            "Command post types: main, tactical, contingency",
            "Staff organized into functional and integrating cells",
            "Cells include current operations, future operations, and plans",
            "Command post operations include shift schedules and SOPs",
        ],
        source_documents=["FM 6-0"],
        category=QueryCategory.TRAP_C,
        hop_count=HopCount.TWO,
        difficulty="medium",
    ),
    GoldAnnotation(
        id="trap_c_07",
        query="What are all the forms of maneuver and when is each one most appropriate?",
        ground_truth_answer=(
            "FM 3-90 (Chapter 2, para 2-12 to 2-25, Table 2-1) describes the five forms of "
            "maneuver: (1) Frontal attack - strikes the enemy across a wide front, used when "
            "enemy is weak or when speed is critical; (2) Penetration - concentrates combat power "
            "at a point to rupture the enemy defense, used against weak spots in enemy defense; "
            "(3) Envelopment - avoids the enemy front to attack flank or rear, used when enemy "
            "flank is assailable; (4) Turning movement - a form of envelopment that forces the "
            "enemy to abandon position, used to force enemy to fight in a new direction; "
            "(5) Infiltration - moves forces through gaps in enemy positions without detection, "
            "used in restrictive terrain or to avoid enemy strength. FM 3-0 (Chapter 6, "
            "Section III) provides operational context for selecting forms of maneuver during "
            "offensive operations."
        ),
        section_references=[
            "FM 3-90, Chapter 2, para 2-12 to 2-25 (Forms of Maneuver)",
            "FM 3-90, Table 2-1 (Forms of Maneuver and Planning Symbols)",
            "FM 3-0, Chapter 6, Section III (Offensive Operations)",
        ],
        information_units=[
            "Frontal attack: strikes across wide front, used when enemy weak or speed critical",
            "Penetration: concentrates at a point to rupture defense, used at weak spots",
            "Envelopment: attacks flank or rear, used when flank is assailable",
            "Turning movement: forces enemy to abandon position and fight in new direction",
            "Infiltration: moves through gaps without detection, used in restrictive terrain",
        ],
        source_documents=["FM 3-90", "FM 3-0"],
        category=QueryCategory.TRAP_C,
        hop_count=HopCount.TWO,
        difficulty="medium",
    ),
    GoldAnnotation(
        id="trap_c_08",
        query="What are all the intelligence capabilities available to a corps during armed conflict?",
        ground_truth_answer=(
            "FM 2-0 (Chapter 7, Section V) describes corps intelligence capabilities including: "
            "the Corps G-2 staff, the corps intelligence cell, the expeditionary military "
            "intelligence brigade (E-MIB), intelligence collection capabilities (Table 7-4), "
            "and all-source intelligence capabilities (Table 7-5). Collection capabilities "
            "include organic HUMINT, SIGINT, GEOINT through UAS, ground-based sensors, and "
            "tactical exploitation. All-source capabilities include the Analysis and Control "
            "Element (ACE). FM 2-0 (Chapter 7, para 7-15 to 7-20) describes how the corps "
            "fights for intelligence during armed conflict. FM 3-0 (para 2-96 to 2-97) "
            "describes the corps as best positioned to achieve convergence with intelligence "
            "and joint capabilities."
        ),
        section_references=[
            "FM 2-0, Chapter 7, Section V (Corps Intelligence)",
            "FM 2-0, Table 7-4 (Corps Collection Capabilities)",
            "FM 2-0, Table 7-5 (Corps All-Source Intelligence Capabilities)",
            "FM 2-0, Chapter 7, para 7-15 to 7-20 (Corps Armed Conflict)",
            "FM 3-0, para 2-96 to 2-97 (Corps role)",
        ],
        information_units=[
            "Corps G-2 staff manages intelligence operations",
            "Corps intelligence cell in the command post",
            "Expeditionary Military Intelligence Brigade (E-MIB) provides organic MI capability",
            "HUMINT collection capabilities",
            "SIGINT collection capabilities",
            "GEOINT through UAS and ground sensors",
            "All-source Analysis and Control Element (ACE)",
            "Corps is best positioned for convergence with joint capabilities (FM 3-0)",
        ],
        source_documents=["FM 2-0", "FM 3-0"],
        category=QueryCategory.TRAP_C,
        hop_count=HopCount.TWO,
        difficulty="hard",
    ),
    GoldAnnotation(
        id="trap_c_09",
        query="What are all the responsibilities of the area of operations owner and where are they defined?",
        ground_truth_answer=(
            "AO responsibilities are defined in FM 3-0 (para 3-138) and FM 3-90 (para 1-56). "
            "Both FMs list the same core responsibilities: (1) terrain management, (2) information "
            "collection, integration, and synchronization, (3) civil affairs operations, "
            "(4) movement control, (5) clearance of fires, (6) security, (7) personnel recovery, "
            "(8) airspace management, and (9) minimum-essential stability tasks which include "
            "establishing civil security and providing immediate needs (food, water, shelter, "
            "medical treatment). FM 3-0 and FM 3-90 both note that commanders can add, remove, "
            "or adjust AO responsibilities based on the situation. FM 3-0 additionally notes "
            "that a land AO does not include airspace control authority by definition."
        ),
        section_references=[
            "FM 3-0, Chapter 3, para 3-138 (AO Responsibilities)",
            "FM 3-90, Chapter 1, para 1-56 (AO Responsibilities)",
        ],
        information_units=[
            "Terrain management",
            "Information collection, integration, and synchronization",
            "Civil affairs operations",
            "Movement control",
            "Clearance of fires",
            "Security",
            "Personnel recovery",
            "Airspace management",
            "Minimum-essential stability: establish civil security",
            "Minimum-essential stability: provide immediate needs (food, water, shelter, medical)",
            "Commanders can adjust responsibilities based on situation",
        ],
        source_documents=["FM 3-0", "FM 3-90"],
        category=QueryCategory.TRAP_C,
        hop_count=HopCount.TWO,
        difficulty="medium",
    ),
    GoldAnnotation(
        id="trap_c_10",
        query="What are all the Army command relationships and what authorities does each one convey?",
        ground_truth_answer=(
            "Army command relationships are described in FM 3-0 (Appendix B, Table B-2), "
            "FM 5-0 (Appendix B, Table B-2), and FM 4-0 (Chapter 2, Table 2-2). The four Army "
            "command relationships are: (1) Organic - assigned to and forming an essential part "
            "of a military organization; (2) Assigned - placed in an organization permanently; "
            "(3) Attached - placed in an organization temporarily; (4) Operational control (OPCON) "
            "- authority to organize and employ commands for specific missions. Each conveys "
            "different authorities for task organization, designation of objectives, and directive "
            "authority. FM 3-0 (Appendix B) and FM 5-0 (Appendix B) also describe support "
            "relationships: direct support, general support, reinforcing, and general "
            "support-reinforcing."
        ),
        section_references=[
            "FM 3-0, Appendix B, Table B-2 (Army Command Relationships)",
            "FM 5-0, Appendix B, Table B-2 (Army Command Relationships)",
            "FM 4-0, Chapter 2, Table 2-2 (Army Command Relationships)",
            "FM 3-0, Appendix B, Table B-3 (Army Support Relationships)",
        ],
        information_units=[
            "Organic: forms essential part of military organization",
            "Assigned: placed permanently in an organization",
            "Attached: placed temporarily in an organization",
            "OPCON: authority to organize and employ for specific missions",
            "Each relationship conveys different task organization authorities",
            "Support relationships: direct support, general support, reinforcing, GS-R",
            "Command relationships described consistently across FM 3-0, FM 5-0, and FM 4-0",
        ],
        source_documents=["FM 3-0", "FM 5-0", "FM 4-0"],
        category=QueryCategory.TRAP_C,
        hop_count=HopCount.THREE,
        difficulty="medium",
    ),

    # =========================================================================
    # CONTROL: Single-Hop Queries (10 queries)
    # Simple factual questions answerable from a single passage
    # =========================================================================
    GoldAnnotation(
        id="control_01",
        query="What are the four steps of actions on contact as described in FM 3-90?",
        ground_truth_answer=(
            "FM 3-90 (Chapter 1, para 1-69 to 1-83, Figure 1-3) describes the four steps of "
            "actions on contact as: (1) React, (2) Develop the situation, (3) Choose an action, "
            "and (4) Execute and report."
        ),
        section_references=["FM 3-90, Chapter 1, para 1-69 to 1-83"],
        information_units=[
            "React",
            "Develop the situation",
            "Choose an action",
            "Execute and report",
        ],
        source_documents=["FM 3-90"],
        category=QueryCategory.CONTROL,
        hop_count=HopCount.ONE,
        difficulty="easy",
    ),
    GoldAnnotation(
        id="control_02",
        query="What are the three types of assigned areas used in the operational framework?",
        ground_truth_answer=(
            "FM 3-0 (Chapter 3, para 3-137) identifies three types of assigned areas: "
            "(1) Area of operations (AO), (2) Zone, and (3) Sector."
        ),
        section_references=["FM 3-0, Chapter 3, para 3-137"],
        information_units=[
            "Area of operations (AO)",
            "Zone",
            "Sector",
        ],
        source_documents=["FM 3-0"],
        category=QueryCategory.CONTROL,
        hop_count=HopCount.ONE,
        difficulty="easy",
    ),
    GoldAnnotation(
        id="control_03",
        query="What are the seven steps of the Military Decision-Making Process?",
        ground_truth_answer=(
            "FM 5-0 (Chapter 5) describes the seven steps of the MDMP as: "
            "(1) Receipt of Mission, (2) Mission Analysis, (3) Course of Action Development, "
            "(4) Course of Action Analysis (War Game), (5) Course of Action Comparison, "
            "(6) Course of Action Approval, (7) Orders Production, Dissemination, and Transition."
        ),
        section_references=["FM 5-0, Chapter 5 (MDMP Overview)"],
        information_units=[
            "Step 1: Receipt of Mission",
            "Step 2: Mission Analysis",
            "Step 3: Course of Action Development",
            "Step 4: Course of Action Analysis (War Game)",
            "Step 5: Course of Action Comparison",
            "Step 6: Course of Action Approval",
            "Step 7: Orders Production, Dissemination, and Transition",
        ],
        source_documents=["FM 5-0"],
        category=QueryCategory.CONTROL,
        hop_count=HopCount.ONE,
        difficulty="easy",
    ),
    GoldAnnotation(
        id="control_04",
        query="What are the four components of the command and control system?",
        ground_truth_answer=(
            "FM 6-0 (Chapter 1, para 1-22 to 1-24, Figure 1-2) identifies the four components "
            "of the C2 system as: (1) People, (2) Processes, (3) Networks, and (4) Command Posts."
        ),
        section_references=["FM 6-0, Chapter 1, para 1-22 to 1-24"],
        information_units=[
            "People",
            "Processes",
            "Networks",
            "Command Posts",
        ],
        source_documents=["FM 6-0"],
        category=QueryCategory.CONTROL,
        hop_count=HopCount.ONE,
        difficulty="easy",
    ),
    GoldAnnotation(
        id="control_05",
        query="What are the four defeat mechanisms described in FM 3-0?",
        ground_truth_answer=(
            "FM 3-0 (Chapter 3, para 3-114 to 3-121) describes the four defeat mechanisms as: "
            "(1) Destroy, (2) Dislocate, (3) Disintegrate, and (4) Isolate."
        ),
        section_references=["FM 3-0, Chapter 3, para 3-114 to 3-121"],
        information_units=[
            "Destroy: apply lethal force so capability can no longer perform its function",
            "Dislocate: obtain positional advantage rendering enemy dispositions less valuable",
            "Disintegrate: disrupt enemy C2 and cohesion of operations",
            "Isolate: separate a force from its sources of support",
        ],
        source_documents=["FM 3-0"],
        category=QueryCategory.CONTROL,
        hop_count=HopCount.ONE,
        difficulty="easy",
    ),
    GoldAnnotation(
        id="control_06",
        query="What is the tactical framework described in FM 3-90?",
        ground_truth_answer=(
            "FM 3-90 (Chapter 1, para 1-32 to 1-42, Figure 1-2) describes the tactical "
            "framework as consisting of four elements: (1) Find the enemy - intel drives fires "
            "and maneuver; (2) Fix the enemy - prevent repositioning or reinforcement; "
            "(3) Finish the enemy - mass combat power to accomplish the mission; (4) Follow "
            "through - defeat in detail, consolidate, reorganize, and transition."
        ),
        section_references=["FM 3-90, Chapter 1, para 1-32 to 1-42"],
        information_units=[
            "Find the enemy: intel drives fires and maneuver",
            "Fix the enemy: prevent repositioning or reinforcement",
            "Finish the enemy: mass combat power to accomplish the mission",
            "Follow through: defeat in detail, consolidate, reorganize, and transition",
        ],
        source_documents=["FM 3-90"],
        category=QueryCategory.CONTROL,
        hop_count=HopCount.ONE,
        difficulty="easy",
    ),
    GoldAnnotation(
        id="control_07",
        query="What are the nine forms of contact described in FM 3-90?",
        ground_truth_answer=(
            "FM 3-90 (Chapter 1, para 1-65) lists nine forms of contact: (1) Direct, "
            "(2) Indirect, (3) Non-hostile, (4) Obstacle, (5) CBRN, (6) Aerial, (7) Visual, "
            "(8) Electromagnetic, and (9) Influence."
        ),
        section_references=["FM 3-90, Chapter 1, para 1-60 to 1-68"],
        information_units=[
            "Direct: ground-based line of sight weapons",
            "Indirect: non-line of sight weapons (artillery, mortars, rockets)",
            "Non-hostile: neutral interactions (civilians, NGOs)",
            "Obstacle: natural and manmade obstacles",
            "CBRN: chemical, biological, radiological, nuclear effects",
            "Aerial: air-based combat platforms",
            "Visual: acquisition via eye or electro-optical systems",
            "Electromagnetic: systems using the EM spectrum",
            "Influence: information dimension interactions",
        ],
        source_documents=["FM 3-90"],
        category=QueryCategory.CONTROL,
        hop_count=HopCount.ONE,
        difficulty="easy",
    ),
    GoldAnnotation(
        id="control_08",
        query="What are the three strategic contexts in which Army forces conduct operations?",
        ground_truth_answer=(
            "FM 3-0 (Introduction, Chapter 1) describes the three strategic contexts as: "
            "(1) Competition below armed conflict, (2) Crisis, and (3) Armed conflict."
        ),
        section_references=["FM 3-0, Chapter 1 and Introduction"],
        information_units=[
            "Competition below armed conflict",
            "Crisis",
            "Armed conflict",
        ],
        source_documents=["FM 3-0"],
        category=QueryCategory.CONTROL,
        hop_count=HopCount.ONE,
        difficulty="easy",
    ),
    GoldAnnotation(
        id="control_09",
        query="What are the seven principles that enable mission command?",
        ground_truth_answer=(
            "FM 6-0 (Chapter 1, para 1-16) lists the seven principles of mission command as: "
            "(1) Competence, (2) Mutual trust, (3) Shared understanding, (4) Commander's intent, "
            "(5) Mission orders, (6) Disciplined initiative, and (7) Risk acceptance."
        ),
        section_references=["FM 6-0, Chapter 1, para 1-15 to 1-18"],
        information_units=[
            "Competence",
            "Mutual trust",
            "Shared understanding",
            "Commander's intent",
            "Mission orders",
            "Disciplined initiative",
            "Risk acceptance",
        ],
        source_documents=["FM 6-0"],
        category=QueryCategory.CONTROL,
        hop_count=HopCount.ONE,
        difficulty="easy",
    ),
    GoldAnnotation(
        id="control_10",
        query="What is the definition of multidomain operations?",
        ground_truth_answer=(
            "FM 3-0 (Chapter 1, para 1-9) defines multidomain operations as the combined arms "
            "employment of joint and Army capabilities to create and exploit relative advantages "
            "to achieve objectives, defeat enemy forces, and consolidate gains on behalf of "
            "joint force commanders (ADP 3-0)."
        ),
        section_references=["FM 3-0, Chapter 1, para 1-9"],
        information_units=[
            "Combined arms employment of joint and Army capabilities",
            "Create and exploit relative advantages",
            "Achieve objectives",
            "Defeat enemy forces",
            "Consolidate gains on behalf of JFCs",
        ],
        source_documents=["FM 3-0"],
        category=QueryCategory.CONTROL,
        hop_count=HopCount.ONE,
        difficulty="easy",
    ),
]


def get_annotations_by_category(category: QueryCategory) -> list[GoldAnnotation]:
    return [a for a in GOLD_ANNOTATIONS if a.category == category]


def get_annotation_by_id(annotation_id: str) -> GoldAnnotation | None:
    for a in GOLD_ANNOTATIONS:
        if a.id == annotation_id:
            return a
    return None


ANNOTATION_STATS = {
    "total": len(GOLD_ANNOTATIONS),
    "trap_a": len(get_annotations_by_category(QueryCategory.TRAP_A)),
    "trap_b": len(get_annotations_by_category(QueryCategory.TRAP_B)),
    "trap_c": len(get_annotations_by_category(QueryCategory.TRAP_C)),
    "control": len(get_annotations_by_category(QueryCategory.CONTROL)),
    "source_documents": [
        "FM 3-0 (Operations, March 2025)",
        "FM 3-90 (Tactics, May 2023)",
        "FM 6-0 (Commander and Staff Organization and Operations, May 2022)",
        "FM 2-0 (Intelligence, October 2023)",
        "FM 5-0 (Planning and Orders Production, November 2024)",
        "FM 4-0 (Sustainment Operations, March 2026)",
    ],
}
