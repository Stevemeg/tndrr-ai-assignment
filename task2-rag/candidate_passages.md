# Candidate source passages: NIST AI 100-1 (AI RMF 1.0)

25 passages spread over the whole document, from the executive summary through Part 1 (risk, trustworthiness), Part 2 (the GOVERN / MAP / MEASURE / MANAGE core and profiles) to Appendices A–D.

* Every passage is copied **verbatim** from the text that `chunking.py` extracts (`extract_pages()`: pypdf text with line-break hyphens re-joined and whitespace collapsed). That is the same text the chunks and `eval_retrieval.py` use. Each passage was checked to occur exactly once in the document.
* **Page** = the PDF page number (1–48, the PDF viewer's page counter), the convention used in `questions.json` and in `ask.py` citations. In the main body, printed "Page N" = PDF page − 5.
* How to use: pick 15. For each, write a question **in your own words**, paste the passage into `source_passage` and the page into `page`. Paraphrased questions, with different words than the passage, are a harder and fairer test of the embeddings than copied keywords.

1. **Page 6** · Executive Summary — what an AI system is

   > The AI RMF refers to an AI system as an engineered or machine-based system that can, for a given set of objectives, generate outputs such as predictions, recommendations, or decisions influencing real or virtual environments. AI systems are designed to operate with varying levels of autonomy (Adapted from: OECD Recommendation on AI:2019; ISO /IEC 22989:2022).

2. **Page 7** · Executive Summary — who AI actors are

   > AI actors are defined by the Organisation for Economic Co-operation and Development (OECD) as “those who play an active role in the AI system lifecycle, including organizations and individuals that deploy or operate AI” [OECD (2019) Artificial Intelligence in Society—OECD iLibrary] (See Appendix A).

3. **Page 9** · 1.1 Understanding and Addressing Risks, Impacts, and Harms — definition of risk

   > In the context of the AI RMF,risk refers to the composite measure of an event’s probability of occurring and the magnitude or degree of the consequences of the corresponding event. The impacts, or consequences, of AI systems can be positive, negative, or both and can result in opportunities or threats (Adapted from: ISO 31000:2018).

   *Note: The missing space in “RMF,risk” is how pypdf extracts this line. Keep it as is.*

4. **Page 10** · 1.2.1 Risk Measurement

   > AI risks or failures that are not well-defined or adequately understood are difficult to measure quantitatively or qualitatively. The inability to appropriately measure AI risks does not imply that an AI system necessarily poses either a high or low risk.

5. **Page 12** · 1.2.2 Risk Tolerance

   > While the AI RMF can be used to prioritize risk, it does not prescribe risk tolerance. Risk tolerance refers to the organization’s or AI actor’s (see Appendix A) readiness to bear the risk in order to achieve its objectives.

6. **Page 13** · 1.2.3 Risk Prioritization — unacceptable risk

   > In cases where an AI system presents unacceptable negative risk levels – such as where significant negative impacts are imminent, severe harms are actually occurring, or catastrophic risks are present – development and deployment should cease in a safe manner until risks can be sufficiently managed.

7. **Page 17** · 3 AI Risks and Trustworthiness — the characteristics

   > Characteristics of trustworthy AI systems include: valid and reliable, safe, secure and resilient, accountable and transparent, explainable and interpretable, privacy-enhanced, and fair with harmful bias managed. Creating trustworthy AI requires balancing each of these characteristics based on the AI system’s context of use.

   *Note: Similar wording (“characteristics of trustworthy AI systems, which include valid and reliable, …”) also appears at the end of p. 7 / start of p. 8.*

8. **Page 17** · 3 AI Risks and Trustworthiness — tradeoffs

   > Addressing AI trustworthiness characteristics individually will not ensure AI system trustworthiness; tradeoffs are usually involved, rarely do all characteristics apply in every setting, and some will be more or less important in any given situation. Ultimately, trustworthiness is a social concept that ranges across a spectrum and is only as strong as its weakest characteristics.

9. **Page 18** · 3.1 Valid and Reliable — validation

   > Validation is the “confirmation, through the provision of objective evidence, that the requirements for a specific intended use or application have been fulfilled” (Source: ISO 9000:2015).

10. **Page 19** · 3.1 Valid and Reliable — robustness

   > Robustness or generalizability is defined as the “ability of a system to maintain its level of performance under a variety of circumstances” (Source: ISO /IEC TS 5723:2022). Robustness is a goal for appropriate system functionality in a broad set of conditions and circumstances, including uses of AI systems not initially anticipated.

11. **Page 20** · 3.3 Secure and Resilient

   > Security and resilience are related but distinct characteristics. While resilience is the ability to return to normal function after an unexpected adverse event, security includes resilience but also encompasses protocols to avoid, protect against, respond to, or recover from attacks.

12. **Page 22** · 3.5 Explainable and Interpretable

   > Transparency can answer the question of “what happened” in the system. Explainability can answer the question of “how” a decision was made in the system. Interpretability can answer the question of “why” a decision was made by the system and its meaning or context to the user.

13. **Page 22** · 3.6 Privacy-Enhanced

   > Privacy refers generally to the norms and practices that help to safeguard human autonomy, identity, and dignity. These norms and practices typically address freedom from intrusion, limiting observation, or individuals’ agency to consent to disclosure or control of facets of their identities (e.g., body, data, reputation).

14. **Page 23** · 3.7 Fair – with Harmful Bias Managed

   > NIST has identified three major categories of AI bias to be considered and managed: systemic, computational and statistical, and human-cognitive. Each of these can occur in the absence of prejudice, partiality, or discriminatory intent.

15. **Page 26** · 5 AI RMF Core — order of the functions

   > Assuming a governance structure is in place, functions may be performed in any order across the AI lifecycle as deemed to add value by a user of the framework. After instituting the outcomes in GOVERN , most users of the AI RMF would start with the MAP function and continue to MEASURE or MANAGE .

   *Note: The space before the commas and the final full stop (“GOVERN ,”, “MANAGE .”) comes from the PDF extraction.*

16. **Page 27** · 5.1 Govern

   > GOVERN is a cross-cutting function that is infused throughout AI risk management and enables the other functions of the process. Aspects of GOVERN , especially those related to compliance or evaluation, should be integrated into each of the other functions.

   *Note: The space before the comma (“GOVERN ,”) comes from the PDF extraction.*

17. **Page 29** · 5.2 Map

   > The MAP function establishes the context to frame risks related to an AI system. The AI lifecycle consists of many interdependent activities involving a diverse set of actors (See Figure 3).

18. **Page 33** · 5.3 Measure

   > The MEASURE function employs quantitative, qualitative, or mixed-method tools, techniques, and methodologies to analyze, assess, benchmark, and monitor AI risk and related impacts. It uses knowledge relevant to AI risks identified in the MAP function and informs the MANAGE function.

19. **Page 36** · 5.4 Manage

   > The MANAGE function entails allocating risk resources to mapped and measured risks on a regular basis and as defined by the GOVERN function. Risk treatment comprises plans to respond to, recover from, and communicate about incidents or events.

20. **Page 37** · Table 4 (MANAGE) — risk response options

   > MANAGE 1.3: Responses to the AI risks deemed high priority, as identified by the MAP function, are developed, planned, and documented. Risk response options can include mitigating, transferring, avoiding, or accepting.

21. **Page 38** · 6 AI RMF Profiles — current vs target profile

   > An AI RMF Current Profile indicates how AI is currently being managed and the related risks in terms of current outcomes. A Target Profile indicates the outcomes needed to achieve the desired or target AI risk management goals. Comparing Current and Target Profiles likely reveals gaps to be addressed to meet AI risk management objectives.

22. **Page 40–41** · Appendix A: AI Actor Tasks — TEVV

   > Ideally, AI actors carrying out verification and validation tasks are distinct from those who perform test and evaluation actions.

   *Note: This sentence starts at the bottom of p. 40 and continues at the top of p. 41. Use page 40 in questions.json; it is a useful test of chunk boundaries.*

23. **Page 43** · Appendix B: How AI Risks Differ from Traditional Software Risks

   > AI systems may require more frequent maintenance and triggers for conducting corrective maintenance due to data, model, or concept drift.

24. **Page 45** · Appendix C: AI Risk Management and Human-AI Interaction

   > Human-AI configurations can span from fully autonomous to fully manual. AI systems can autonomously make decisions, defer decision making to a human expert, or be used by a human decision maker as an additional opinion. Some AI systems may not require human oversight, such as models used to improve video compression.

25. **Page 47** · Appendix D: Attributes of the AI RMF

   > Be outcome-focused and non-prescriptive. The Framework should provide a catalog of outcomes and approaches rather than prescribe one-size-fits-all requirements.
