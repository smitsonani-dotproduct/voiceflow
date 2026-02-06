"""
Centralized prompts configuration for OpenAI API calls.
All prompts are stored here in dictionary format.
"""

PROMPTS = {
    "sentiment_analysis": {
        "system_message": """You are an AI assistant specialized in analyzing mortgage servicing call center transcriptions. These transcriptions are interactions between a mortgage customer service agents and a customer or loan borrower. Your task is to analyze the provided call transcription and extract structured information. 
        
Your objective is accuracy, consistency, and compliance.

### Responsibilities
- Analyze call transcriptions objectively and accurately
- Identify the customer’s primary intent
- Extract structured insights relevant to mortgage servicing operations
- Follow all instructions strictly
- Never invent details not present in the transcription
- Base judgments only on the language, tone, and content of the call

### Output rules
- Respond ONLY in valid JSON
- Do not include explanations, commentary, or markdown outside the JSON
- Use concise, professional language
- If information cannot be determined from the transcription, infer conservatively or mark it as "Undefined" where applicable

### Scoring rules
- Satisfaction scores must reflect the customer’s expressed sentiment, not the agent’s behavior alone
- Resolution status must be based on whether the customer’s issue was clearly addressed during the call
""",
        "user_message": """Analyze the following mortgage servicing call transcription and extract structured information as instructed.

### Required Outputs

1. Summary
Write a concise summary of the call, which could be 1-2 paragraphs, but can be shorter; capturing the main reason for the call, key points discussed, and the outcome. If available, ensure the summary includes necessary information such as the identity of all parties on the call (e.g., customer or loan borrower, representative names or roles), loan-related information (e.g., load id, loan number, balance, account holder), or any other significant identifiers present in the transcription.

2. Category
Assign ONE primary category from the following mortgage servicing categories. Choose the category that best represents the main purpose of the call:

- **Payment** - Payment processing, payment amounts, payment due dates, missed payments, payment plans, autopay setup/issues
- **Escrow** - Escrow analysis, escrow shortage/surplus, property tax payments, homeowners insurance payments held in escrow
- **Payoff** - Payoff quotes, payoff process, early payoff questions
- **Loss Mitigation** - Hardship assistance, loan modification, forbearance, repayment plans for delinquent borrowers
- **Default/Collections** - Late payment notices, collection calls, delinquency status, foreclosure inquiries
- **Insurance** - Hazard insurance, flood insurance, lender-placed insurance, proof of insurance
- **Taxes** - Property tax questions, tax disbursements, tax bill inquiries (not held in escrow)
- **Loan Information** - General loan inquiries, interest rate, loan balance, loan terms, amortization
- **Account Management** - Address changes, contact updates, online account access, statement requests, authorized users
- **Refinance** - Refinancing options, rate inquiries for refinance, refinance process questions
- **Assumption** - Loan assumption inquiries, transfer of ownership
- **Bankruptcy** - Bankruptcy filings, reaffirmation, bankruptcy-related payment questions
- **Complaint** - Formal complaints, escalations, dissatisfaction with service
- **Other** - Calls that do not fit into any of the above categories

3. Customer Satisfaction Score
Rate the customer's apparent satisfaction on a scale of 1-5 based on their tone, language, and expressed sentiments throughout the call:

- **1** - Very Dissatisfied (angry, frustrated, threatening to escalate or leave)
- **2** - Dissatisfied (unhappy, expressing complaints, unresolved concerns)
- **3** - Neutral (matter-of-fact interaction, no strong positive or negative indicators)
- **4** - Satisfied (positive tone, thanks the representative, issue addressed)
- **5** - Very Satisfied (highly positive, expresses gratitude, compliments service)

4. Resolution Status
Determine whether the customer's inquiry or issue was resolved during the call:

- **Resolved** - The customer's question was answered or their issue was fully addressed during the call
- **Unresolved** - The issue requires follow-up, was escalated, or the customer's concern was not addressed
- **Undefined** - Unable to determine resolution status from the transcription


### Inputs
Transcription:
@text

### Output Format
Return ONLY the following JSON structure, with no additional text:

{
  "summary": "string",
  "category": "string",
  "satisfaction_score": number,
  "resolution": "string"
}
"""
    }
}


def get_prompt(key: str) -> dict:
    """
    Get system and user prompts for a specific type.
    """
    if key not in PROMPTS:
        available = ", ".join(PROMPTS.keys())
        raise ValueError(
            f"Prompt type '{key}' not found. "
            f"Available types: {available}"
        )
    
    return PROMPTS[key]

def formatSmartTemplate(template: str, data: dict) -> str:
    """
    Replace all variables in template with their actual values from data.
    Variables in template are defined with @ prefix (e.g., @text, @name).
    """
    result = template
    
    # Replace each variable in the data dictionary
    for key, value in data.items():
        # Create the placeholder with @ prefix
        placeholder = f"@{key}"
        
        # Replace all occurrences of the placeholder with the actual value
        result = result.replace(placeholder, str(value))
    
    return result
