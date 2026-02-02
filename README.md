# CONVODRIFT: A Multi-Turn Conversational Dataset for Modeling Stylistic Tone Evolution

**ConvoDrift** is a dataset designed to model progressive conversational tone drift under fixed semantic intent. It addresses the limitation of existing NLP benchmarks that often treat style as a static property or focus on single-turn transfers. ConvoDrift explicitly models how stylistic tone evolves across multi-turn interactions.

This repository contains the codebase and datasets described in the paper:  
*CONVODRIFT: A Multi-Turn Conversational Dataset for Modeling Stylistic Tone Evolution*

## Overview

The dataset consists of two core components:

1.  **ConvoDrift-Conversation:** A multi-turn conversational dataset capturing progressive stylistic tone drift.
2.  **ConvoDrift-RLHF:** A persona-conditioned pairwise preference dataset constructed from the conversational trajectories to enable personalized and pluralistic alignment.

## Dataset Structure

The repository is organized into the following main directories:

### 1. Conversational Dataset (`Conversational_dataset/`)

This directory contains the core multi-turn conversations.

*   **Location:** `Conversational_dataset/conversational_data/`
*   **Genres:** The data is stratified across 5 communication genres:
    *   `business_emails/`
    *   `casual_emails/`
    *   `linkedin_posts/`
    *   `quotes_wishes/`
    *   `tweets/`
*   **Format:** JSONL
*   **Structure:** Each conversation consists of 6 prompt-response pairs.
    *   **Drift Label (`Drift`):** Binary (`True`/`False`). Indicates if the user prompt explicity requested a tone change.
    *   **Direction Label (`Direction`):** Ternary label indicating the direction of stylistic shift:
        *   `0`: No stylistic change.
        *   `1`: Shift towards **Casual / Expressive** tone.
        *   `2`: Shift towards **Formal / Precise** tone.

### Data Distribution

| Communication Genre | Scenario | Count |
| :--- | :--- | :--- |
| **business email** | Client Meeting & Sales | 138 |
| | Project Status / Deliverables | 138 |
| | Vendor Negotiation / Procurement | 138 |
| | Recruitment & HR | 134 |
| | Customer Success / Support | 132 |
| | Finance & Invoicing | 132 |
| | Legal & Compliance | 132 |
| | Partnership / Business Development | 132 |
| | Marketing Campaigns | 132 |
| | Product Feedback & Release Notes | 132 |
| | Operations & Logistics | 132 |
| | Executive / Board Updates | 132 |
| | Investor Relations | 132 |
| | IT & Security | 131 |
| | Training & Enablement | 130 |
| | Cross-Functional Collaboration | 50 |
| | Change Management | 50 |
| | Corporate Social Responsibility | 50 |
| | Crisis Communication | 50 |
| | Learning & Development | 50 |
| | Diversity & Inclusion | 50 |
| | Performance Management | 50 |
| | Corporate Announcements | 50 |
| | Innovation & Strategy | 50 |
| | Team Recognition & Culture | 50 |
| | Workplace Policies | 50 |
| | Internal Operations | 50 |
| | Research & Insights | 50 |
| | Client Onboarding | 50 |
| | Post-Event Follow-Up | 50 |
| | Budget & Planning | 50 |
| | Employee Wellbeing | 50 |
| | Knowledge Sharing | 50 |
| | Cross-Border Collaboration | 50 |
| | Stakeholder Engagement | 50 |
| **casual email** | Friendship & Catching Up | 204 |
| | Family & Relatives | 204 |
| | Travel & Vacations | 204 |
| | Celebrations & Events | 200 |
| | Apologies & Reconciliations | 198 |
| | Invitations | 198 |
| | Sharing News or Updates | 198 |
| | Hobbies & Interests | 198 |
| | Casual Advice / Tips | 198 |
| | Thank You & Appreciation | 197 |
| | College & Campus Life | 65 |
| | Weekend Plans | 65 |
| | Sports & Fitness | 65 |
| | Neighborhood & Community | 65 |
| | Random Curiosities | 65 |
| | Dreams & Future Plans | 65 |
| | Funny Stories & Jokes | 64 |
| | Music & Movies | 64 |
| | Food & Dining | 64 |
| | Pets & Animals | 64 |
| | Life Reflections | 63 |
| | Fashion & Style | 60 |
| | Tech & Gadgets | 60 |
| | Childhood Memories | 60 |
| | Books & Learning | 60 |
| | Nature & Outdoors | 60 |
| | Friendship Memories | 60 |
| | Daily Routines | 60 |
| | Weather & Seasons | 59 |
| | Late-Night Thoughts | 59 |
| **linkedin post** | Work-Life Balance & Wellness | 204 |
| | Learning Communities & Peer Groups | 204 |
| | Sustainability & Green Initiatives | 204 |
| | Cross-Cultural Experiences | 200 |
| | Alumni & Education Journeys | 198 |
| | Career Transitions & Pivots | 198 |
| | Freelancing & Independent Work | 198 |
| | Recognition & Awards | 198 |
| | Employee Advocacy & Brand Ambassadorship | 198 |
| | Future of Work & Skills | 198 |
| | Cross-Disciplinary Work | 80 |
| | Storytelling in Business | 79 |
| | Resilience & Mindset | 78 |
| | Early Career Experiences | 78 |
| | Workplace Creativity | 78 |
| | Ethics & Integrity | 78 |
| | Talent Development | 78 |
| | Decision Making | 78 |
| | Global Trends | 77 |
| | Failures & Lessons Learned | 77 |
| | Future Aspirations | 75 |
| | Emotional Intelligence | 75 |
| | Role Models & Inspirations | 75 |
| | Career Breaks & Comebacks | 72 |
| | Networking Etiquette | 72 |
| **quotes/wishes** | Teacher Appreciation Quotes | 170 |
| | Retirement Wishes | 170 |
| | New Baby Congratulations | 170 |
| | Travel & Adventure Quotes | 170 |
| | Environmental & Nature Quotes | 170 |
| | Patriotic & National Day Messages | 170 |
| | Literary & Book Lover Quotes | 170 |
| | Music & Art Inspiration Quotes | 170 |
| | Mindfulness & Peace Quotes | 169 |
| | Housewarming Wishes | 168 |
| | Sports & Team Spirit Quotes | 166 |
| | Encouragement for Challenges | 165 |
| | Leadership & Vision Quotes | 165 |
| | Innovation & Creativity Quotes | 165 |
| | Volunteering & Kindness Quotes | 165 |
| | Spiritual & Faith-Based Messages | 165 |
| | Seasonal Greetings | 164 |
| | Welcome Messages | 163 |
| | Milestone Celebration Wishes | 163 |
| | Resilience & Strength Quotes | 162 |
| **tweet** | Everyday Reflections | 102 |
| | Creativity & Ideas | 102 |
| | Work-Life Balance | 102 |
| | Mentorship & Guidance | 102 |
| | Risk & Courage | 102 |
| | Human Behavior | 102 |
| | Change & Adaptability | 102 |
| | Small Wins | 102 |
| | Dreams & Ambitions | 102 |
| | Conflict & Resolution | 102 |
| | Curiosity & Wonder | 102 |
| | Values & Principles | 102 |
| | Serendipity | 102 |
| | Growth Mindset | 98 |
| | Influence & Impact | 96 |
| | Resilience & Recovery | 96 |
| | Time & Priorities | 96 |
| | Unlearning & Rethinking | 96 |
| | Moments of Joy | 96 |
| | Critical Thinking | 95 |
| | Personal Growth | 51 |
| | Workplace Humor | 51 |
| | Tech & Innovation | 51 |
| | Leadership Insights | 51 |
| | Daily Motivation | 51 |
| | Remote Work Life | 51 |
| | Failures & Comebacks | 51 |
| | Networking Moments | 51 |
| | Learning & Curiosity | 51 |
| | Cultural Perspectives | 51 |
| | Future of Work | 51 |
| | Life Hacks | 51 |
| | Public Speaking | 49 |
| | Industry Hot Takes | 48 |
| | Entrepreneurship | 48 |
| | Collaboration & Teamwork | 48 |
| | Decision Fatigue | 48 |
| | Generational Shifts | 48 |
| | Social Impact | 47 |
| | Random Observations | 47 |

### 2. RLHF Dataset (`RLHF_dataset/`)

This directory contains preference data for Reinforcement Learning from Human Feedback (RLHF), conditioned on specific personas.

*   **Location:** `RLHF_dataset/main_labeled_personas/`
*   **Files:**
    *   `labeled_Main_final_dataset_Persona_A.jsonl`
    *   `labeled_Main_final_dataset_Persona_B.jsonl`
    *   `labeled_Main_final_dataset_Persona_C.jsonl`
    *   `labeled_Main_final_dataset_Persona_D.jsonl`
    *   `labeled_Main_final_dataset_Persona_E.jsonl`

#### Personas
The RLHF dataset uses 5 distinct personas to model pluralistic preferences:

| Persona | Description |
| :--- | :--- |
| **Task-First** | Focuses on efficiency, clarity, and structured reasoning. Prioritizes actionable outcomes. |
| **Relationship-First** | Emphasizes warmth, empathy, and cooperative language. |
| **Authority-First** | Communicates with confidence, decisiveness, and assertive tone. |
| **Calm & Careful** | Uses measured, precise language to minimize risk and ensure safety. |
| **Expressive & Energetic** | Engages with enthusiasm, vivid language, and excitement. |

## Methodology

### Data Generation
*   **Conversation Generation:** Used **GPT-4.1-mini** to generate consistent, quality multi-turn conversations where prompts progressively request stylistic refinements while preserving semantic intent.
*   **Drift/Direction Labeling:** Used **GPT-5-nano** to annotate explicit drift and direction labels.

### Evaluation
The dataset has undergone comprehensive evaluation including:
*   **Human Validation:** Using three annotators to validate drift annotations and conversation quality.
*   **LLM-as-a-Judge:** Using models like Gemini and Claude to validate label accuracy.
*   **Automatic Metrics:** Semantic similarity (SBERT) and lexical similarity (ROUGE-L, Jaccard) to ensure meaning is preserved while style changes.
