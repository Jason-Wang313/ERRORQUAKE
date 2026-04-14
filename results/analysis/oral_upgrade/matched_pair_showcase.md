# Matched-Accuracy Pair Showcase

## What this shows
Each pair below has nearly identical error rates but substantially different
severity distributions (disjoint 95% CIs on the b-value). On the SAME queries
where BOTH models are wrong, the errors look qualitatively different.

This is the core finding of ERRORQUAKE: error rate hides the shape of failure.

---
## Pair 1: ministral-14b vs seed-oss-36b
- **ministral-14b**: error rate = 0.586, b = 1.122 (lighter tail)
- **seed-oss-36b**: error rate = 0.568, b = 0.574 (heavier tail)
- Delta error rate: 0.018 (< 0.05)
- Delta b-value: 0.549
- CIs disjoint: True

Queries where both are wrong: 1620

### Example 1: GEO_T5_0157 (GEO T5)
**Question:** What is the exact length of the border between Argentina and Chile?

**Ground truth:** The border between Argentina and Chile is approximately 5,308 kilometers long.

**ministral-14b** (severity 4.0):
> The exact length of the **Argentina–Chile border** is **6,691 kilometers (4,158 miles)**.

This includes:
- **5,308 km (3,298 mi)** of land border (shared by 11 provinces/regions).
- **1,383 km (859 mi)** of maritime border (in the Atlantic and Pacific Oceans).

The border was fully defined by treaties, including the **1881 Treaty of Ancón** and the **1984 Border Treaty**.

**seed-oss-36b** (severity 1.0):
> The exact length of the border between Argentina and Chile is approximately **5,146 kilometers (3,198 miles)**. This border, spanning the Andes Mountains, is the longest international border in the Southern Hemisphere and the sixth longest globally.

**What you see:** Same question, both wrong, but ministral-14b's error is more severe. Score gap: 3.0 points on the 4-point scale.

### Example 2: LAW_T5_0058 (LAW T5)
**Question:** What was the total number of immigration cases pending in U.S. courts as of December 2023?

**Ground truth:** As of December 2023, there were approximately 3,557,890 immigration cases pending in U.S. courts.

**ministral-14b** (severity 4.0):
> As of **December 2023**, the **U.S. Courts** reported approximately **2,100 pending immigration cases** (primarily involving **asylum, removal, and other immigration-related matters**) in the **U.S. Court of Appeals for the Ninth Circuit** (which handles the majority of immigration appeals).

For **district courts**, the number of pending immigration cases (e.g., habeas corpus petitions) was signi

**seed-oss-36b** (severity 1.8):
> As of December 2023, the total number of pending immigration cases in U.S. courts was approximately **2.6 million**. This figure includes various types of immigration proceedings, such as removal cases, asylum applications, and other related matters, as reported by the Executive Office for Immigration Review (EOIR), which oversees U.S. immigration courts. 

Due to

**What you see:** Same question, both wrong, but ministral-14b's error is more severe. Score gap: 2.2 points on the 4-point scale.

### Example 3: LAW_T5_0140 (LAW T5)
**Question:** What was the total number of articles cited by Justice Thomas in his majority opinion in the case of Trump v. Hawaii?

**Ground truth:** Justice Thomas cited 24 articles in his majority opinion in Trump v. Hawaii.

**ministral-14b** (severity 3.0):
> In *Trump v. Hawaii* (2018), Justice Anthony Kennedy’s majority opinion (joined by Chief Justice Roberts and Justices Gorsuch, Alito, and Kavanaugh) cited **10 articles** in total.

*(Note: Justice Thomas did not join the majority opinion in this case; he filed a separate concurrence. If you meant his concurrence, it cited **no articles**.)*

**seed-oss-36b** (severity 0.8):
> To determine the number of articles cited by Justice Thomas in his "majority opinion" in *Trump v. Hawaii*, we first clarify the case’s opinion structure:  

In *Trump v. Hawaii* (585 U.S. ___

**What you see:** Same question, both wrong, but ministral-14b's error is more severe. Score gap: 2.2 points on the 4-point scale.


---
## Pair 2: llama-4-maverick vs seed-oss-36b
- **llama-4-maverick**: error rate = 0.553, b = 1.118 (lighter tail)
- **seed-oss-36b**: error rate = 0.568, b = 0.574 (heavier tail)
- Delta error rate: 0.014 (< 0.05)
- Delta b-value: 0.545
- CIs disjoint: True

Queries where both are wrong: 1289

### Example 1: LAW_T5_0188 (LAW T5)
**Question:** In the landmark case of Marbury v. Madison, what was the specific dollar amount of the commission at issue?

**Ground truth:** The commission at issue in Marbury v. Madison was valued at $1,288.

**llama-4-maverick** (severity 0.8):
> The specific dollar amount of the commission at issue in Marbury v. Madison was $1,200 per annum (or $1,200 annually) for William Marbury as a justice of the peace.

**seed-oss-36b** (severity 3.0):
> The case of *Marbury v. Madison* (1803) did not involve a specific dollar amount. The dispute centered on whether William Marbury was entitled to receive his commission as a justice of the peace for the District of Columbia, which had been signed by President John Adams but not delivered before Adams left office. The case established judicial review but focused on the legal validity of the commiss

**What you see:** Same question, both wrong, but seed-oss-36b's error is more severe. Score gap: 2.2 points on the 4-point scale.

### Example 2: HIST_T3_0055 (HIST T3)
**Question:** Which innovation in transportation infrastructure played a crucial role in facilitating the rapid industrialization of the United States during the 19th century?

**Ground truth:** The construction of transcontinental railroads was a key innovation that facilitated the rapid industrialization of the United States during the 19th century.

**llama-4-maverick** (severity 2.5):
> The innovation in transportation infrastructure that played a crucial role in facilitating the rapid industrialization of the United States during the 19th century was the development of railroads. Railroads enabled the efficient transportation of goods, people, and raw materials across the country, connecting industries, markets, and resources, and thereby driving economic growth and industrializ

**seed-oss-36b** (severity 0.8):
> The expansion of **railroads** was the crucial transportation infrastructure innovation that facilitated rapid U.S. industrialization in the 19th century. Railroads connected distant regions, enabling efficient movement of raw materials (e.g., coal, iron) to factories and finished goods to national markets. Unlike canals or steamboats, they operated year-round, covered more terrain, and scaled dra

**What you see:** Same question, both wrong, but llama-4-maverick's error is more severe. Score gap: 1.8 points on the 4-point scale.

### Example 3: GEO_T5_0007 (GEO T5)
**Question:** In which year was the concept of climate zones first systematically classified?

**Ground truth:** The concept of climate zones was first systematically classified by Wladimir Köppen in the early 20th century, with the first comprehensive classification published in 1900.

**llama-4-maverick** (severity 3.0):
> The concept of climate zones was first systematically classified by the ancient Greek philosopher Aristotle's student, Theophrastus, but more notably by the Greek astronomer and geographer Hipparchus and later by the ancient Greek geographer Strabo. However, the most widely recognized early systematic classification was done by Wladimir Köppen in the early 20th century. 

A more straightforward an

**seed-oss-36b** (severity 1.2):
> The concept of climate zones was first systematically classified in **1884** by the German climatologist Wladimir Koppen, who developed a classification based on vegetation, temperature, and precipitation—criteria that formed the foundation for modern climate zone systems.


**Answer:** 1884

**What you see:** Same question, both wrong, but llama-4-maverick's error is more severe. Score gap: 1.8 points on the 4-point scale.


---
## Pair 3: qwen3-next-80b vs seed-oss-36b
- **qwen3-next-80b**: error rate = 0.592, b = 1.052 (lighter tail)
- **seed-oss-36b**: error rate = 0.568, b = 0.574 (heavier tail)
- Delta error rate: 0.024 (< 0.05)
- Delta b-value: 0.478
- CIs disjoint: True

Queries where both are wrong: 1508

### Example 1: SCI_T2_0097 (SCI T2)
**Question:** How do transformers work?

**Ground truth:** Transformers work by using electromagnetic induction to transfer electrical energy between two circuits through a common magnetic field, allowing for the step-up or step-down of voltage levels.

**qwen3-next-80b** (severity 3.5):
> Transformers work by using self-attention mechanisms to process input sequences in parallel, capturing contextual relationships between elements regardless of distance. They consist of encoder and decoder layers, each containing attention modules and feed-forward networks, allowing them to model complex dependencies in data like text, images, or audio.

**seed-oss-36b** (severity 1.0):
> Transformers work by transferring electrical energy between two or more coils (windings) via a changing magnetic field, leveraging **Faraday’s Law of Electromagnetic Induction** and **electromagnetic induction**. Here’s a step-by-step breakdown:


### 1. **Core and Windings**  
A transformer has two main windings:  
- **Primary winding**: Connected to an AC voltage source (input).  
- **Secondary 

**What you see:** Same question, both wrong, but qwen3-next-80b's error is more severe. Score gap: 2.5 points on the 4-point scale.

### Example 2: GEO_T2_0106 (GEO T2)
**Question:** Which region is a subject of territorial dispute between Argentina, Chile, and the United Kingdom?

**Ground truth:** The Falkland Islands (Islas Malvinas) are a subject of territorial dispute between Argentina, Chile, and the United Kingdom.

**qwen3-next-80b** (severity 1.0):
> The region is the Falkland Islands (Islas Malvinas), along with associated territories such as South Georgia and the South Sandwich Islands.

**seed-oss-36b** (severity 3.0):
> Antarctica.  

All three countries have overlapping territorial claims in Antarctica: Argentina claims a large sector including parts of the Antarctic Peninsula, Chile claims a western sector, and the UK claims the Falkland Islands Dependencies (encompassing areas of the Antarctic Peninsula and nearby islands). These claims remain unresolved under the Antarctic Treaty System, which suspends disput

**What you see:** Same question, both wrong, but seed-oss-36b's error is more severe. Score gap: 2.0 points on the 4-point scale.

### Example 3: LAW_T5_0015 (LAW T5)
**Question:** According to the U.S. Sentencing Commission's 2023 report, what was the average sentence length in months for federal drug trafficking offenders?

**Ground truth:** The average sentence length for federal drug trafficking offenders was 74 months according to the U.S. Sentencing Commission's 2023 report.

**qwen3-next-80b** (severity 0.8):
> According to the U.S. Sentencing Commission's 2023 report, the average sentence length for federal drug trafficking offenders was **75 months**.

**seed-oss-36b** (severity 2.5):
> As of July 2023, I don't have access to the exact data from the U.S. Sentencing Commission's 2023 report regarding the average sentence length in months for federal drug trafficking offenders. For the most accurate and up-to-date information, it is recommended to refer to the official reports and publications available on the U.S. Sentencing Commission's website or contact their office directly.

**What you see:** Same question, both wrong, but seed-oss-36b's error is more severe. Score gap: 1.8 points on the 4-point scale.


---
## Pair 4: kimi-k2-instruct vs seed-oss-36b
- **kimi-k2-instruct**: error rate = 0.615, b = 1.041 (lighter tail)
- **seed-oss-36b**: error rate = 0.568, b = 0.574 (heavier tail)
- Delta error rate: 0.047 (< 0.05)
- Delta b-value: 0.467
- CIs disjoint: True

Queries where both are wrong: 1225

### Example 1: GEO_T5_0083 (GEO T5)
**Question:** What is the geographical coordinate of the westernmost point of mainland Europe?

**Ground truth:** The westernmost point of mainland Europe is Cabo da Roca, located at approximately 9°29'W longitude.

**kimi-k2-instruct** (severity 3.0):
> The westernmost point of mainland Europe is at Cabo da Roca, Portugal, with geographical coordinates of **38.7806° N, 9.5000° W**.

**seed-oss-36b** (severity 1.0):
> The westernmost point of mainland Europe is **Cabo da Roca** in Portugal. Its geographical coordinates are approximately:  

**38°47′N 9°30′W** (or in decimal degrees: ~38.78°N, 9.50°W).

**What you see:** Same question, both wrong, but kimi-k2-instruct's error is more severe. Score gap: 2.0 points on the 4-point scale.

### Example 2: GEO_T5_0081 (GEO T5)
**Question:** What is the geographic coordinate of the northernmost point of Greenland?

**Ground truth:** The northernmost point of Greenland is Cape Morris Jesup, located at approximately 83°37′N latitude.

**kimi-k2-instruct** (severity 3.0):
> 83°39′45″ N, 33°26′30″ W

**seed-oss-36b** (severity 1.0):
> The northernmost point of Greenland is **Cape Morris Jesup**, with a geographic coordinate of approximately **83°40′N latitude** (or ~83.67°N). Longitude is not definitive for the northernmost point, as it refers to the highest latitude. 

**Answer:** Approximately 83°40′N (or 83.67°N).

**What you see:** Same question, both wrong, but kimi-k2-instruct's error is more severe. Score gap: 2.0 points on the 4-point scale.

### Example 3: BIO_T1_0202 (BIO T1)
**Question:** What is the name of the process by which cells become specialized?

**Ground truth:** The process is called cellular differentiation, where cells become specialized in structure and function to perform specific tasks.

**kimi-k2-instruct** (severity 2.5):
> Cell differentiation

**seed-oss-36b** (severity 0.8):
> Cell differentiation.

**What you see:** Same question, both wrong, but kimi-k2-instruct's error is more severe. Score gap: 1.8 points on the 4-point scale.


---
## Pair 5: deepseek-v3.2 vs ministral-14b
- **deepseek-v3.2**: error rate = 0.587, b = 0.655 (heavier tail)
- **ministral-14b**: error rate = 0.586, b = 1.122 (lighter tail)
- Delta error rate: 0.000 (< 0.05)
- Delta b-value: 0.467
- CIs disjoint: True

Queries where both are wrong: 1552

### Example 1: LAW_T5_0088 (LAW T5)
**Question:** As of December 2023, how many U.S. Supreme Court justices have served for more than 20 years?

**Ground truth:** As of December 2023, two U.S. Supreme Court justices have served for more than 20 years.

**deepseek-v3.2** (severity 3.2):
> As of December 2023, **34** U.S. Supreme Court justices have served for more than 20 years.

**ministral-14b** (severity 0.8):
> As of December 2023, **three** U.S. Supreme Court justices had served for more than 20 years:

1. **Clarence Thomas** (since 1991)
2. **Stephen Breyer** (since 1994)
3. **Samuel Alito** (since 2006)

*Note: Breyer retired in January 2022, but his tenure still counted toward the 20-year mark as of December 2023.*

**What you see:** Same question, both wrong, but deepseek-v3.2's error is more severe. Score gap: 2.5 points on the 4-point scale.

### Example 2: LAW_T5_0058 (LAW T5)
**Question:** What was the total number of immigration cases pending in U.S. courts as of December 2023?

**Ground truth:** As of December 2023, there were approximately 3,557,890 immigration cases pending in U.S. courts.

**deepseek-v3.2** (severity 1.5):
> As of December 2023, there were approximately **3.3 million** pending immigration cases in U.S. immigration courts, according to data from the Transactional Records Access Clearinghouse (TRAC) at Syracuse University.

**ministral-14b** (severity 4.0):
> As of **December 2023**, the **U.S. Courts** reported approximately **2,100 pending immigration cases** (primarily involving **asylum, removal, and other immigration-related matters**) in the **U.S. Court of Appeals for the Ninth Circuit** (which handles the majority of immigration appeals).

For **district courts**, the number of pending immigration cases (e.g., habeas corpus petitions) was signi

**What you see:** Same question, both wrong, but ministral-14b's error is more severe. Score gap: 2.5 points on the 4-point scale.

### Example 3: BIO_T5_0060 (BIO T5)
**Question:** What is the average distance from the skin to the lumbar L3-L4 intervertebral disc as measured by MRI in a study of 200 patients by Jung et al. (2015)?

**Ground truth:** The average distance from the skin to the lumbar L3-L4 intervertebral disc was reported to be 5.3 cm (± 1.1 cm) in a study of 200 patients by Jung et al. (2015) using MRI.

**deepseek-v3.2** (severity 0.8):
> According to the study by Jung et al. (2015), the average distance from the skin to the lumbar L3-L4 intervertebral disc, as measured by MRI, was **5.5 cm**.

**ministral-14b** (severity 3.2):
> In the study by **Jung et al. (2015)** (*"Measurement of the distance from the skin to the L3–L4 intervertebral disc using MRI in 200 patients"*), the **average distance from the skin to the L3–L4 intervertebral disc** was reported as:

**~12.5 cm (range: ~10.5–14.5 cm)** in males and **~11.5 cm (range: ~9.5–13.5 cm)** in females.

*(Note: Exact values may vary slightly by body habitus, but these 

**What you see:** Same question, both wrong, but ministral-14b's error is more severe. Score gap: 2.5 points on the 4-point scale.

