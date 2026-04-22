# Sample Queries And Outputs

Captured on 2026-04-22 using:

```bash
python scripts/test_query.py "<question>" --json
```

## Query 1

Question: What are the rights of domestic workers in Hong Kong?

Chatbot response:
Based on retrieved Labour Department context, domestic worker rights include rest days, statutory holidays, paid annual leave, wage protection, employer-provided accommodation/food terms, medical treatment coverage during employment, and return travel obligations under contract conditions. The response also states that employers should not compel work on rest days/statutory holidays and notes that some entitlements depend on continuous-contract rules.

## Query 2

Question: What are the rules for recruitment agencies?

Chatbot response:
Based solely on retrieved snippets, the assistant says it is unsure of the full private recruitment agency rule set because top matches were service-index pages about Labour Department employment services rather than detailed recruitment-agency regulations. It explicitly flags missing details such as licensing, fee, and conduct standards in the retrieved evidence.

## Query 3

Question: What is the statutory minimum wage in Hong Kong?

Chatbot response:
The assistant cites that Hong Kong SMW is an hourly rate, with HKD 42.1 per hour effective 1 May 2025, and references approved increase information to HKD 43.1 per hour effective 1 May 2026 from retrieved Labour Department legislation/FAQ sources.

## Query 4

Question: 如果僱主沒有給我休息日，我可以怎麼做？

Chatbot response:
根據檢索到的資料，回覆建議可向勞工處求助，並指出僱主無合理辯解而不給予休息日可構成違法（最高罰款 50,000 元）。回覆同時標示目前片段未提供完整投訴程序，建議直接聯絡勞工處進一步跟進。

## Known Limitation

Questions needing precise cross-ordinance reasoning can still surface partial or imperfectly matched evidence when top-k retrieval overweights nearby FAQ/service snippets.

