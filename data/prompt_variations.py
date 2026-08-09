MATH_PROMPT = lambda question, reasoning, answer, language: f"""In {language}, generate a NEW, ORIGINAL math problem that is AS DIFFICULT as the reference below. Do NOT copy, paraphrase, or reuse it in any way.

Reference (difficulty calibration only — do not reproduce):
<question>
"{question}"
</question>

<reasoning>
{reasoning}
</reasoning>

<answer>
{answer}
</answer>

Requirements for your generated problem:
- The question, reasoning, and answer should all be in {language}.
- Requires non-trivial reasoning steps (no single-step shortcuts)
- Draws from: number theory, combinatorics, algebra, geometry, or probability
- Is self-contained and precisely stated
- Reasoning should be in includes a complete step-by-step derivation
- Answer includes just the final result

IMPORTANT: generate a (question, reasoning, answer) triplet; wrap your question, reasoning, and answer in the following special tokens:
<question> Insert your {language} question here. </question>
<reasoning> Insert the thinking and general reasoning here in {language}. </reasoning>
<answer> Insert your short {language} answer here. </answer>"""






STEM_PROMPT = lambda question, reasoning, answer, language: f"""In {language}, generate a NEW, ORIGINAL STEM multiple-choice question (MCQA) that is AS DIFFICULT as the reference below. Do NOT copy, paraphrase, or reuse it in any way.

Reference (difficulty calibration only — do not reproduce):
<question>
{question}
</question>

<reasoning>
{reasoning}
</reasoning>

<answer>
{answer}
</answer>

Requirements for your generated question:
- The question, reasoning, and answer should all be in {language}.
- Draws from STEM domains: physics, chemistry, biology, computer science, engineering, or mathematics
- Requires non-trivial conceptual or quantitative reasoning (no single-step lookups or trivial recall)
- Has exactly 4 answer choices labeled (A), (B), (C), (D) — only one is correct
- Distractors are plausible and reflect common misconceptions or near-miss reasoning errors
- Is self-contained, unambiguous, and precisely stated
- Reasoning walks through the correct derivation/justification step by step and explains why each distractor is wrong
- Answer is the correct letter only, e.g. "(B)"

IMPORTANT: generate a (question, reasoning, answer) triplet; wrap them in the following special tokens:
<question> Insert your {language} question stem followed by the four answer choices (A)–(D). </question>
<reasoning> Insert the step-by-step reasoning in {language}, including why each distractor is incorrect. </reasoning>
<answer> Insert only the correct letter, e.g. "(A)". </answer>"""







CHAT_PROMPT = lambda question, reasoning, answer, language: f"""In {language}, generate a NEW, ORIGINAL instruction-following task that is AS COMPLEX as the reference below. Do NOT copy, paraphrase, or reuse it in any way.

Reference (complexity calibration only — do not reproduce):
<question>
"{question}"
</question>

<reasoning>
{reasoning}
</reasoning>

<answer>
{answer}
</answer>

Requirements for your generated task:
- The question, reasoning, and answer should all be in {language}.
- Instruction imposes at least as many explicit constraints as the reference (e.g. format, length, style, content restrictions, conditional logic)
- Constraints are specific and verifiable — a reader can check whether the response satisfies each one
- Draws from: open-ended knowledge tasks (Alpaca-style), conversational requests (LMArena-style), or format-constrained tasks (IFEval-style)
- Instruction is self-contained and unambiguous
- Reasoning walks through how each constraint is satisfied, step by step
- Answer is a complete response that fully obeys every constraint in the instruction

IMPORTANT: generate a (question, answer) pair; wrap your question and answer in the following special tokens:
<question> Insert your {language} instruction here. </question>
<reasoning> Insert the step-by-step reasoning in {language}. </reasoning>
<answer> Insert the complete response in {language} that satisfies all constraints. </answer>"""
