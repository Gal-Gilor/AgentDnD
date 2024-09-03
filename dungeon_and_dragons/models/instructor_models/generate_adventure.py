from typing import Literal

from pydantic import BaseModel, Field


class Quest(BaseModel):
    """Craft Detailed and Immersive Quests:

    Design each quest with precision, ensuring it is highly detailed, creative, and cohesive. Quests should be compelling and well-integrated into the adventure's narrative, offering unique objectives that engage the players and enhance the story. Use vivid language that emphasizes action verbs and descriptive adjectives to create a personalized and memorable quest experience.

    **Instructions:**
    - **Quest Description:** Provide a detailed and imaginative description of the quest, clearly defining its objective, challenges, motivation, and potential outcomes.
    - **Type:** Specify whether the quest is a primary or secondary objective, ensuring it aligns with the overall theme and progression of the adventure.
    """

    quest: str = Field(..., description="The Quest of the Adventure")
    type: Literal["Primary", "Secondary"] = Field(
        ...,
        description=(
            "Primary Objectives: The main goals of the adventure.\n"
            "Secondary Objectives: Optional goals that provide "
            "additional rewards or challenges."
        ),
    )


class Location(BaseModel):
    """Design Immersive and Detailed Location:

    Define each area with a high level of detail, creativity, and cohesion, ensuring it enhances the overall adventure experience. Areas should be compelling and well-integrated into the narrative, providing unique environments for encounters and exploration. Use vivid language that focuses on action verbs and descriptive adjectives to create an immersive setting that personalizes the adventure for players.

    **Instructions:**
    - **Title:** Provide a descriptive title for the area, capturing its unique essence within the adventure.
    - **Location:** Specify the exact location of the area, including its name, geographic context, strategic importance, and access points.
    - **Surroundings:** Offer a vivid description of the area's surroundings, detailing the environment, atmosphere, and any terrain features that could influence the encounter.
    """

    title: str = Field(
        ...,
        description="A descriptive title or name of the area, helping to distinguish it from other locations in the adventure.",
    )
    location: str = Field(
        ...,
        description=(
            "Specify the exact location where the encounter takes place:\n"
            "- **Name:** Provide the name of the location if it's known or significant.\n"
            "- **Geographic Context:** Mention the region, terrain, or nearby landmarks that help place the location within the world.\n"
            "- **Strategic Importance:** Describe any strategic or narrative reasons why this location is relevant to the encounter.\n"
            "- **Access and Entry Points:** Detail how the players reach this location, including any obstacles or notable paths."
        ),
    )
    surroundings: str = Field(
        ...,
        description=(
            "A vivid, highly detailed description of the area's surroundings, providing:\n"
            "- **Environment:** The physical setting (e.g., forest, cave, dungeon room) and its key features.\n"
            "- **Atmosphere:** Vivid sensory details (what the characters see, hear, smell, and feel) to immerse players.\n"
            "- **Terrain:** Any environmental factors that could influence the experience, such as narrow corridors, obstacles, or hazards.\n"
            "Ensure the description immerses the players in the scene and helps them understand their surroundings and potential interactions."
        ),
    )


class Encounter(BaseModel):
    """Design Adventure Encounters with Precision and Creativity:

    Craft each encounter with a high level of detail, creativity, and cohesion to ensure it contributes meaningfully to the overall narrative. Encounters should be compelling and well-thought-out, each offering a unique challenge that engages the players and drives the story forward. Use vivid language that incorporates action verbs and descriptive adjectives to bring each encounter to life, personalizing the adventure for the players.

    **Instructions:**
    - **Encounter Description:** Provide a comprehensive description of the encounter, including its name, situation, objectives, key elements, and possible outcomes.
    - **Type:** Define the nature of the encounter, whether it's combat, social interaction, or exploration, ensuring it aligns with the overall theme of the adventure.
    - **Quest:** Describe the quest associated with the encounter, focusing on the objectives, challenges, rewards, and clues that guide players toward completion.
    - **Location:** Specify the exact location of the encounter, including the name, geographic context, strategic importance, and access points.
    - **Surroundings:** Detail the setting of the encounter with a vivid description that includes the environment, atmosphere, and any terrain features that could influence the encounter.
    """

    encounter: str = Field(
        ...,
        description=(
            "A comprehensive description of the encounter, including:\n"
            "- **Name:** A descriptive title for the encounter, particularly if it's significant or recurring.\n"
            "- **Situation:** The event or challenge that players will face, such as a battle, puzzle, or social interaction.\n"
            "- **Objectives:** Clearly outline the goals for the players, such as defeating enemies, solving a puzzle, or negotiating with NPCs.\n"
            "- **Key Elements:** Highlight any important details like specific enemies, traps, or NPCs that are central to the encounter.\n"
            "- **Outcome:** Possible results of the encounter, including success, failure, and any consequences or rewards."
        ),
    )
    type: Literal["Combat", "Social", "Exploration"] = Field(
        ...,
        description=(
            "The nature of the encounter:\n"
            "- Combat: A battle or skirmish with enemies or creatures.\n"
            "- Social: An interaction or negotiation with NPCs or other characters.\n"
            "- Exploration: A scenario involving puzzles, traps, or obstacles that require creative problem-solving."
        ),
    )
    quest: Quest = Field(
        ...,
        description=(
            "An imaginative, highly detailed description of the quest, including:\n"
            "- **Objective:** The specific goal or task the players must complete.\n"
            "- **Challenges:** Any obstacles, enemies, or hazards that stand in the way of success.\n"
            "- **Rewards:** The potential benefits or consequences of completing the quest.\n"
            "- **Clues:** Any hints, tips, or leads that guide players toward the quest's completion."
        ),
    )
    details: Location = Field(
        ...,
        description=(
            "A highly detailed information about the encounter's location, including its title, location, and surroundings."
        ),
    )


class Antagonist(BaseModel):

    antagonist_name: str = Field(
        ...,
        description=(
            "The name of the main antagonist in the adventure. "
            "Choose a name that reflects the antagonist's character, origins, or role in the story."
        ),
    )
    background: str = Field(
        ...,
        description=(
            "A compelling origin story for the main antagonist that seamlessly "
            " aligns with the adventure's theme and objectives. "
            "The background should include:\n"
            "- **Motivation**: What drives the antagonist? (Greed, revenge, power, etc.)\n"
            "- **Personality**: Highlight their strengths, weaknesses, and unique quirks.\n"
            "- **Goals**: Define what the antagonist aims to achieve.\n"
        ),
    )


class Theme(BaseModel):
    """Define the Core Theme of the Adventure with Creativity and Clarity:

    The theme sets the tone and direction for the entire adventure. It must be meticulously crafted to be both compelling and cohesive, providing a strong foundation for the story. Use vivid, descriptive language that captures the essence of the adventure and resonates with the players. The theme should be well-thought-out, reflecting the narrative's depth and engaging the target audience.

    **Instructions:**
    - **Title:** Choose a title that captures the adventure's essence and intrigues players. It should reflect the campaign's mood, hint at the overall theme, and create intrigue without revealing too much.
    - **Theme:** Clearly define the adventure's theme, whether it's mystery, personal growth, conflict, or exploration. This theme should permeate every aspect of the adventure, providing a unifying thread that ties the story together.
    - **Target Audience:** Specify the intended audience, whether the adventure is beginner-friendly or geared towards experienced players. Tailor the adventure's complexity, challenges, and tone to suit this audience.
    - **Length:** Determine the adventure's length, whether it's a one-shot, short adventure, or full campaign. The scope of the adventure should be well-aligned with its theme and objectives.
    - **Average Party Level:** Set the average party level for the characters, ensuring it matches the adventure's challenges and the experience level of the target audience.
    """

    title: str = Field(
        ...,
        description=(
            "The Title of the Adventure.\n"
            "Choose a title that captures the essence of the adventure and intrigues players."
            "Instructions for Naming a D&D Campaign:\n"
            "- Choose words that match the campaign's mood (e.g., dark, mysterious, adventurous).\n"
            "- Ensure the title hints at the overall theme.\n"
            "- Include elements of the main conflict or quest.\n"
            "- Focus on what the players will be striving to achieve.\n"
            "- Feature a significant artifact, location, or character.\n"
            "- Capture the essence of the setting in the title.\n"
            "- Create intrigue by being suggestive rather than explicit.\n"
            "- Avoid revealing too much of the story in the title.\n"
            "- Aim for a simple, catchy title that's easy to remember.\n"
            "- Ensure the title is unique and stands out."
        ),
    )
    theme: Literal["Mystery", "Personal Growth", "Conflict", "Exploration"] = Field(
        ..., title="Theme", description=("The Theme of the Adventure\n")
    )
    target_audience: Literal["Beginner-Friendly", "Experienced"] = Field(
        ...,
        description=(
            "The target audience for the adventure.\n"
            "**Beginner-Friendly**: Simple puzzles, straightforward combat, clear objectives."
            "**Experienced**: Complex puzzles, challenging encounters, multiple paths "
            "and choices."
        ),
    )
    length: Literal["One-Shot", "Short", "Campaign"] = Field(
        ...,
        description=(
            "The length of the adventure.\n"
            "One-Shot: A single session of play.\n"
            "Short Adventure: 2-3 sessions.\n"
            "Campaign: A series of interconnected adventures.\n"
        ),
    )
    average_party_level: int = Field(
        ...,
        description=(
            "The Average Party Level of the Characters.\n"
            "For a beginner-friendly adventure, the APL should be 1-3.\n"
            "For an experienced adventure, the APL should be 4-6."
        ),
    )
    reward_for_success: str = Field(
        ...,
        description=(
            "Reward for Success: Design a reward that reflects the significance of the players' achievements.\n"
            "The reward should align with the adventure's theme and provide a sense of accomplishment. Consider various types of rewards, such as magical items, unique artifacts, character growth opportunities, or narrative advancements.\n"
            "- **Impactful Rewards:** Ensure the reward feels significant and proportionate to the challenges faced by the players.\n"
            "- **Types of Rewards:** Include magical items, unique artifacts, or other meaningful benefits that enhance the adventure and characters.\n"
            "- **Enhanced Experience:** Use rewards to reinforce the adventure's themes and provide additional motivation for players, enriching the story and overall experience."
        ),
    )


class Background(BaseModel):
    """Develop a Rich and Engaging Background for the Adventure:

    The background should be a meticulously detailed and creatively crafted narrative that sets the stage for the entire adventure. It must be cohesive and compelling, drawing players into the world and motivating them to engage with the story. Use vivid, action-oriented language to create a personalized experience that resonates with the players and enhances the adventure's depth.

    **Instructions:**
    - **Backstory:** Establish a detailed historical context, highlighting key events, figures, and the catalyst that sets the adventure in motion.
    - **The Hook:** Create a compelling event or situation that draws the characters into the adventure, clearly establishing the stakes and aligning with the adventure’s theme.
    - **The Inciting Incident:** Describe the pivotal event that propels the characters into the core of the adventure, introducing central conflicts and challenges.
    - **The Stakes:** Clearly define the consequences of success or failure, emphasizing the urgency and importance of the characters' actions.
    - **The Twist:** Plan a surprising and impactful turn of events that challenges the characters and alters the course of the adventure, maintaining narrative cohesion.
    - **Antagonist:** Develop a rich origin story and name for the main antagonist, detailing their motivations, personality, and how they pose a formidable challenge to the players.
    - **Encounters:** Include a diverse list of encounters that test the characters' abilities and contribute meaningfully to the narrative, each crafted with creativity and relevance to the story's objectives.
    """

    backstory: str = Field(
        ...,
        description=(
            "Establish the Backstory:\n"
            "- **Historical Context**: Lay the groundwork by describing the history, legends, or ancient events that have shaped the world. Highlight key moments that directly influence the present.\n"
            "- **Catalyst for the Adventure**: Identify the specific event or series of events that trigger the adventure. This could be a recent discovery, a rising threat, or a call to action that draws the characters into the story.\n"
            "- **Key Figures and Factions**: Introduce important characters, organizations, or factions whose actions have led to the current situation. Explain their motives and how they intersect with the players' journey.\n"
            "- **Setting the Tone**: Use the backstory to set the tone of the adventure—whether it’s a tale of heroism, tragedy, mystery, or epic conflict. The prologue should create anticipation and intrigue, drawing players into the world and the upcoming challenges."
        ),
    )
    the_hooks: str = Field(
        ...,
        description=(
            "One or two compelling events or situations that draw the characters into the adventure, bringing them together as a party. These hooks should be designed to intrigue and engage the adventurers, who do not know each other at the beginning. The events should be seemingly random but effective in forming the group.\n"
            "- **Initial Engagement:** Create an event or situation that immediately captures the characters' attention and compels them to act. This could be a mysterious message, a sudden attack, or an urgent plea for help. Be imaginative and ensure the hook effectively prompts the characters to come together.\n"
            "- **Setting the Stakes:** Clearly define the stakes involved in the hook, such as saving lives, preventing disaster, or uncovering a hidden truth. The stakes should be significant enough to motivate the characters to join forces.\n"
            "- **Alignment:** Ensure the hook aligns with the adventure’s theme and objectives, setting the stage for the story's tone and direction. The hook should seamlessly introduce the characters to the adventure and the overarching plot.\n"
            "- **Party Formation:** Since the adventurers do not know each other at the start, the hook should facilitate their coming together in a plausible manner, creating a natural reason for them to unite and embark on the adventure."
        ),
    )
    the_inciting_incident: str = Field(
        ...,
        description=(
            "The event that sets the plot in motion.\n"
            "- **Triggering Event**: Identify the pivotal moment that propels the characters "
            "from their ordinary lives into the heart of the adventure. This could be an "
            "unexpected betrayal, the discovery of a powerful artifact, or the eruption "
            "of a long-dormant threat.\n"
            "- **Irreversible Action**: Ensure the inciting incident is a decisive event that "
            "leaves the characters with no option but to move forward. It should create a "
            "sense of urgency and drive the narrative.\n"
            "- **Introduce Key Conflicts**: Use this moment to introduce the central conflict or "
            "challenge that will shape the adventure.\n"
            "- **Alignment**: The inciting incident must align with the adventure's theme "
            "and objectives, establishing a clear path forward that resonates with the "
            "overarching story."
        ),
    )
    the_stakes: str = Field(
        ..., description="The consequences of failure or success for the characters."
    )
    the_twist: str = Field(
        description=(
            "A surprising turn of events that changes the course of the adventure."
            "- **Unexpected Revelation**: Plan a twist that defies the characters' expectations, "
            "turning the adventure on its head. This could be the betrayal of a trusted ally, "
            "the revelation of a hidden villain, or the discovery of a deeper conspiracy.\n"
            "- **Changing the Stakes**: The twist should significantly alter the course of the "
            "adventure, raising the stakes and forcing the characters to rethink their plans. "
            "It should challenge their assumptions and push them into uncharted territory.\n"
            "- **Maintaining Cohesion**: Ensure the twist aligns with the overall narrative "
            "and theme, even as it surprises the players. It should feel like a natural, "
            "if unforeseen, development within the story.\n"
        )
    )
    antagonist: Antagonist = Field(
        ...,
        description=(
            "A compelling origin story and name for the main antagonist that  "
            "seamlessly aligns with the adventure's theme and objectives.\n"
            "Instructions for Creating a D&D antagonist:\n"
            "- **Origin Story**: Develop a rich backstory for the main antagonist "
            "that seamlessly integrates with the adventure's theme and objectives. "
            "Explain how their past experiences, ambitions, or traumas have shaped "
            "them into the villain they are today.\n"
            "- **Motivation**: Clearly define what drives the antagonist—whether it's "
            "a thirst for power, revenge, greed, or something more complex. Their "
            "motivation should be understandable, even if it's not justifiable.\n"
            "- **Personality and Traits**: Flesh out the antagonist's personality, "
            "including their strengths, weaknesses, and quirks. Consider how these "
            "traits influence their actions and decisions throughout the adventure.\n"
            "- **Name Creation**: Choose a name that reflects the antagonist's character, "
            "origins, or role in the story.\n"
        ),
    )
    encounters: list[Encounter] = Field(
        ...,
        description=(
            "A list of at least 10 encounters that the characters might face throughout the adventure.\n"
            "The encounters should be diverse, and aligned with the adventure's theme and objectives.\n"
            "Instructions for Creating D&D Encounters:\n"
            "- **Combat Encounters**: Plan battles against enemies or creatures that challenge "
            "the characters' combat skills and tactics. Include a mix of foes with different "
            "abilities and strengths to keep the combat engaging.\n"
            "- **Social Encounters**: Design interactions with NPCs or other characters that "
            "require diplomacy, persuasion, or deception. Create memorable personalities and "
            "dialogue that reflect the encounter's objectives.\n"
            "- **Exploration Encounters**: Develop puzzles, traps, or obstacles that test the "
            "characters' problem-solving abilities and creativity. Include clues, hints, and "
            "rewards that encourage exploration and discovery.\n"
        ),
    )


class CreateAdventure(Background, Theme):
    """Design Deeply Engaging and Immersive Adventures:

    Act as an experienced Dungeon Master, crafting adventures that are highly creative, compelling, cohesive, and well-thought-out. Your task is to write detailed, vivid descriptions that bring the adventure to life. Each section should be rich in action verbs and descriptive adjectives, providing at least three paragraphs of engaging content.

    **Instructions:**
    - **Narrative Development:**
        - **Immersion:** Use vivid language to fully immerse players in the setting and story.
        - **Setting the Scene:** Describe the environment, atmosphere, and surroundings in detail, making the world feel alive.
        - **Character Interaction:** Create dynamic dialogue for characters and NPCs that reflects their personalities and motives.
        - **Plot Design:** Outline a clear and engaging plot with a strong beginning, middle, and end.
        - **Conflict & Resolution:** Introduce significant conflicts and ensure satisfying resolutions.
        - **Thematic Cohesion:** Maintain consistent themes and tone throughout the adventure.
        - **Objective Alignment:** Align quests and objectives with the main narrative and theme.
        - **Innovative Elements:** Incorporate original ideas, twists, and challenges to make the adventure memorable.
        - **Character Development:** Facilitate character growth and evolution.
        - **Interactive Challenges:** Develop puzzles, traps, and encounters that encourage creative thinking.
        - **Player Agency:** Ensure player choices have meaningful consequences and shape the story.
        - **Lore and History:** Integrate rich lore and history to add depth to the world.
        - **Cultural Details:** Include cultural nuances and societal structures to enhance realism.
        - **Worldbuilding:** Provide detailed descriptions of geography, culture, politics, and technology, creating a vibrant, dynamic world.

    **Story Elements:**
    1. **Backstory:** Establish the historical context, key events, and influential figures. Highlight the catalyst for the adventure and set the tone.
    2. **The Hook:** Introduce an engaging event or situation that draws characters into the adventure, capturing their attention and establishing the stakes.
    3. **The Inciting Incident:** Describe the pivotal moment that thrusts characters into the adventure’s core, introducing central conflicts and aligning with the story's progression.
    4. **The Stakes:** Define the consequences of success or failure, emphasizing the urgency and importance of the characters' actions.
    5. **The Twist:** Incorporate a surprising turn of events that challenges the characters and alters the adventure's course, adding depth while maintaining narrative cohesion.
    6. **The Antagonist:** Develop a compelling antagonist with a rich origin story, clear motivations, and a personality that aligns with the adventure’s theme, presenting a formidable challenge.
    7. **Encounters:**
        - **Diverse and Varied:** Design a range of encounters—combat, social, and exploration—that test the characters and contribute to the narrative. Avoid repetition by including different creatures, traps, and puzzles.
        - **Theme and Setting:** Ensure encounters fit the adventure's theme and setting.
        - **Rewards and Consequences:** Provide meaningful rewards for success and consequences for failure.
        - **Player Agency and Choice:**
            - **Branching Paths:** Offer choices that impact the story’s direction with multiple solutions to challenges.
            - **Meaningful Decisions:** Ensure player choices alter outcomes, the story, or the world.
            - **Roleplaying Opportunities:** Create moments for characters to use their skills and backgrounds in interactions with NPCs.
    8. **Worldbuilding:**
        - **Immersive Details:** Describe locations, NPCs, creatures, and items with evocative language and sensory details.
        - **Lore and History:** Develop a rich background including history, culture, events, and factions.
        - **Unique Elements:** Introduce specific elements unique to your world, such as creatures, magic systems, or cultural traditions.
    9. **Compelling Story:**
        - **Strong Hook:** Engage players from the start with an intriguing hook that draws them into the story.
        - **Clear Motivation:** Provide a compelling reason for players to embark on the adventure.
        - **Conflict and Stakes:** Introduce conflict and escalate stakes to keep players invested in the outcome.
    """


class ImprovedAdventure(Background, Theme):
    """Edit an existing adventure to enhance its narrative, challenges, and thematic elements.
    As an experienced Dungeon Master, your task is to transform this D&D adventure into a more immersive, engaging, and cohesive story. Ensure each aspect of the adventure is well-thought-out and rich in detail. Use vivid, descriptive language to create an engaging world that feels alive and dynamic. Each description should be at least 3 paragraphs long, with detailed, evocative language that brings scenes, characters, and settings to life.

    **If the adventure involves a mystery:**
    - Provide additional clues and improve logical reasoning to make the mystery more engaging and solvable. Ensure that clues are well-integrated and lead players to a satisfying resolution.

    Avoid generic tropes and ensure the narrative is unique and compelling. Enhance the story's depth and complexity, making sure that every element contributes meaningfully to the overall adventure. Include at least 6 diverse encounters that align with the adventure's theme and objectives. Make sure each encounter is thoroughly described and contributes to the story's progression, offering both challenges and rewards.

    **Instructions:**
    - **Detailed Descriptions:** Provide thorough, multi-paragraph descriptions for each element of the adventure, using vivid and descriptive language.
    - **Engagement:** Ensure that the narrative is engaging and avoids generic tropes.
    - **Mystery Enhancements:** If applicable, add more clues and refine the logical progression of the mystery.
    - **Encounters:** Include at least 6 well-defined encounters that are varied and thematically appropriate.
    """
