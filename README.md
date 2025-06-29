# Project ML Tutor

Dwight is going to be an AI assistant using RAG to help you learn concept much faster. Learning content refreshers! matched with lecture notes, discussion worksheets, and midterm questions.

## Setup

1. `pip install -r requirements.txt`
2. Create a `.env` file with `OPENAI_API_KEY=<your key>`

---

## 1. Ingest & Preprocess the Documents

1. **Document Types**

   * **Lecture Notes** (concise definitions, algorithm sketches)
   * **Midterm Questions** (hard, no solutions)
   * **Midterm Solutions** (answers + explanations)
   * **Discussion Worksheets** (medium difficulty Q/A)
   * **Small-Group Tutoring Transcripts** (easy Q/A)

2. **Parsing & Chunking**

   * Break each document into small “knowledge chunks” (e.g. per definition, per theorem, per Q/A pair).
   * For messy OCR, run each raw page through `src/ingest/llm_chunk.py` which
     uses GPT to clean up the text and split it into 3–5 coherent chunks.
   * For each chunk, record metadata:

     * **Source** (lecture, midterm-question, solution, etc.)
     * **Topic Tags** (e.g. “Disjoint Sets,” “Weighted Union,” “Path Compression”)
     * **Difficulty** (easy/medium/hard based on source or by length/complexity heuristics).

3. **Topic Tagging (Precompute)**

   * Use an LLM or simple keyword classifier to assign each chunk one or more topic tags.
   * Example: a question “Why does path compression improve amortized cost?” → tags `{DisjointSets, PathCompression}`.

4. **Embedding Generation**

   * Run each chunk through a sentence- or paragraph-embedding model (e.g. OpenAI’s text-embedding-ada).
   * Store embeddings + metadata in a fast vector database (FAISS, Pinecone, Chroma, etc.).

---

## 2. Build the Retrieval-Augmented Generation (RAG) Layer

1. **Student Query → Embedding**

   * When the student asks “Explain weighted quick‐union,” embed that query.

2. **Vector Lookup**

   * Find the top-k chunks whose embeddings have highest cosine similarity to the query.
   * You might retrieve, say, two definitions from lecture notes, one relevant Q/A from solutions, and one example from group discussions—covering both formal definition and applied intuition.

3. **Context Assembly**

   * Assemble those top-k chunks (with their source and difficulty metadata) into a prompt for the generative model.

4. **Answer Generation**

   * The LLM uses that context to generate:

     * A clear definition.
     * A step‐by‐step walkthrough of the algorithm.
     * A simple example (drawn or paraphrased from discussion worksheets).
     * A harder exercise (drawn from midterms) as a challenge.

---

## 3. Adaptive Tutoring Strategies

1. **Scaffolded Explanations**

   * If the student signals “I don’t get it,” the agent can regenerate the explanation using only “easy” chunks (filter by difficulty ≤ medium) or switch to more visuals/analogies.

2. **Active Recall & Spaced Repetition**

   * Automatically generate flashcards from key Q/A pairs.
   * Schedule review prompts: e.g., “Tomorrow: Why does path compression yield near-constant time?”

3. **Knowledge Tracing**

   * Track which topics the student has practiced (via quizzes/exercises).
   * Maintain a per-topic “mastery score” and bias future retrieval toward under-mastered topics.

4. **Dynamic Difficulty Adjustment**

   * If the student easily answers a challenge, the agent ups the difficulty (next midterm problem).
   * If they struggle, it pulls in more “medium” or “easy” examples and/or re-explains fundamentals.

---

## 4. Categorization & Analytics (Precomputed Work)

* **Topic Coverage Map**

  * For each topic (e.g. Disjoint Sets), record how many chunks exist at each difficulty from each source.
  * Helps visualize “you’ve seen 5 definitions, 3 examples, 2 hard problems.”

* **Question Clusters**

  * Cluster similar midterm questions (using embedding distance) so the agent can avoid near-duplicates as practice.

* **Exercise Bank**

  * Tag every question by topic and difficulty.
  * Enables “give me three practice problems on Path Compression at medium difficulty.”

---

## 5. Putting It All Together: The Tutoring Session

1. **Student:** “Can you explain path compression?”
2. **Agent:**

   1. Retrieves a succinct definition from lecture notes.
   2. Pulls an illustrative example from small-group Q/A.
   3. Summarizes the amortized-cost proof from midterm solutions.
   4. Offers a self-check: “Try to trace 8→2→0 under path compression. What tree do you end up with?”
3. **Student attempts** and answers.
4. **Agent:**

   * If correct, “Great! Next, here’s a medium problem from a past midterm.”
   * If incorrect, “Let’s walk it through step by step using this example from our discussion worksheets.”

---

## 6. Why This Works

* **RAG + Vector DB** lets the agent ground every explanation in *your* materials.
* **Metadata** (topic, difficulty, source) ensures explanations are at the right level.
* **Precomputed tagging & clustering** speeds up retrieval and reduces cold-start overhead.
* **Adaptive strategies** (scaffold, spaced repetition, dynamic difficulty) mimic best practices in human tutoring.

---

### Next Steps

1. **Prototype the ingestion pipeline** (parse, tag, embed, index).
2. **Choose tooling**: LangChain or Haystack for RAG orchestration; Pinecone/FAISS/Chroma for vector store.
3. **Define topic taxonomy** for your course outline.
4. **Implement a minimal chat interface** that loops student query → retrieval → generation → feedback.

With this architecture, the AI “learns” from your lecture notes, midterms, solutions, and discussion transcripts—and then tutors *just like you’d want* at 2 AM before the final.
