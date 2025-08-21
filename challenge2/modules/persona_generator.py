"""
Persona Generator Module
Generates diverse loan defaulter personalities for testing voice agents
"""

import random
from dataclasses import dataclass, asdict
from typing import List, Optional
from enum import Enum

class TestScenario(Enum):
    """Different test scenarios for debt collection"""
    FIRST_CONTACT = "first_contact"
    PAYMENT_REMINDER = "payment_reminder" 
    HARDSHIP_CASE = "hardship_case"
    DISPUTE_RESOLUTION = "dispute_resolution"
    PAYMENT_ARRANGEMENT = "payment_arrangement"
    AGGRESSIVE_CUSTOMER = "aggressive_customer"
    CONFUSED_CUSTOMER = "confused_customer"
    PROMISE_TO_PAY = "promise_to_pay"

@dataclass
class DebtorPersona:
    """Represents a loan defaulter personality for testing"""
    name: str
    age: int
    debt_amount: float
    days_past_due: int
    income: float
    employment_status: str
    personality_traits: List[str]
    communication_style: str
    payment_history: str
    financial_stress_level: int  # 1-10 scale
    negotiation_likelihood: float  # 0-1 probability
    preferred_payment_method: str
    previous_interactions: int
    
    def to_dict(self):
        return asdict(self)

class PersonaGenerator:
    """Generates diverse loan defaulter personalities for testing"""
    
    def __init__(self):
        self.personality_traits = [
            "anxious", "defensive", "cooperative", "hostile", "confused", 
            "apologetic", "demanding", "evasive", "honest", "manipulative",
            "overwhelmed", "prideful", "suspicious", "understanding", "volatile"
        ]
        
        self.communication_styles = [
            "aggressive", "passive", "assertive", "passive-aggressive", 
            "collaborative", "withdrawn", "emotional", "logical"
        ]
        
        self.employment_statuses = [
            "unemployed", "part-time", "full-time", "self-employed", 
            "retired", "disabled", "student", "contractor"
        ]
        
        self.payment_histories = [
            "first_time_default", "chronic_defaulter", "occasional_late", 
            "recent_hardship", "payment_plan_violator", "good_until_recently"
        ]
        
        self.payment_methods = [
            "bank_transfer", "debit_card", "cash", "money_order", 
            "payment_plan", "partial_payments", "asset_liquidation"
        ]
    
    def generate_persona(self, scenario: TestScenario = None) -> DebtorPersona:
        """Generate a realistic debtor persona"""
        
        # Generate basic demographics
        age = random.randint(22, 75)
        debt_amount = round(random.uniform(500, 50000), 2)
        days_past_due = random.randint(30, 365)
        income = round(random.uniform(0, 80000), 2)
        
        # Select random characteristics
        traits = random.sample(self.personality_traits, random.randint(2, 4))
        comm_style = random.choice(self.communication_styles)
        employment = random.choice(self.employment_statuses)
        payment_history = random.choice(self.payment_histories)
        payment_method = random.choice(self.payment_methods)
        
        # Calculate derived attributes
        stress_level = min(10, max(1, int(debt_amount / income * 10) if income > 0 else 10))
        negotiation_likelihood = self._calculate_negotiation_likelihood(traits, comm_style, stress_level)
        
        # Generate name
        first_names = ["Alex", "Jordan", "Taylor", "Casey", "Morgan", "Jamie", "Riley", "Drew"]
        last_names = ["Smith", "Johnson", "Williams", "Brown", "Davis", "Miller", "Wilson", "Moore"]
        name = f"{random.choice(first_names)} {random.choice(last_names)}"
        
        return DebtorPersona(
            name=name,
            age=age,
            debt_amount=debt_amount,
            days_past_due=days_past_due,
            income=income,
            employment_status=employment,
            personality_traits=traits,
            communication_style=comm_style,
            payment_history=payment_history,
            financial_stress_level=stress_level,
            negotiation_likelihood=negotiation_likelihood,
            preferred_payment_method=payment_method,
            previous_interactions=random.randint(0, 5)
        )
    
    def _calculate_negotiation_likelihood(self, traits: List[str], comm_style: str, stress_level: int) -> float:
        """Calculate likelihood of successful negotiation based on persona attributes"""
        base_likelihood = 0.5
        
        # Adjust based on traits
        trait_adjustments = {
            "cooperative": 0.2, "honest": 0.15, "understanding": 0.1,
            "hostile": -0.3, "evasive": -0.2, "manipulative": -0.15,
            "defensive": -0.1, "anxious": -0.05
        }
        
        for trait in traits:
            base_likelihood += trait_adjustments.get(trait, 0)
        
        # Adjust based on communication style
        style_adjustments = {
            "collaborative": 0.2, "assertive": 0.1, "passive": 0.05,
            "aggressive": -0.25, "passive-aggressive": -0.15, "withdrawn": -0.1
        }
        base_likelihood += style_adjustments.get(comm_style, 0)
        
        # Adjust based on stress level (higher stress = lower negotiation likelihood)
        stress_penalty = (stress_level - 5) * 0.02
        base_likelihood -= stress_penalty
        
        return max(0.0, min(1.0, base_likelihood))

    def generate_batch(self, count: int, scenarios: List[TestScenario] = None) -> List[DebtorPersona]:
        """Generate multiple personas for testing"""
        personas = []
        for i in range(count):
            scenario = random.choice(scenarios) if scenarios else None
            persona = self.generate_persona(scenario)
            personas.append(persona)
        return personas


# Example usage and testing (you can remove this section if not needed)
if __name__ == "__main__":
    # Test the persona generator
    generator = PersonaGenerator()
    
    print("🧪 Testing Persona Generator")
    print("=" * 50)
    
    # Generate a few sample personas
    test_personas = generator.generate_batch(3)
    
    for i, persona in enumerate(test_personas, 1):
        print(f"\n{i}. {persona.name}")
        print(f"   Age: {persona.age}, Debt: ${persona.debt_amount:,.2f}")
        print(f"   Days Past Due: {persona.days_past_due}")
        print(f"   Income: ${persona.income:,.2f}")
        print(f"   Employment: {persona.employment_status}")
        print(f"   Traits: {', '.join(persona.personality_traits)}")
        print(f"   Communication: {persona.communication_style}")
        print(f"   Payment History: {persona.payment_history}")
        print(f"   Stress Level: {persona.financial_stress_level}/10")
        print(f"   Negotiation Likelihood: {persona.negotiation_likelihood:.2f}")
        print(f"   Preferred Payment: {persona.preferred_payment_method}")
        print(f"   Previous Interactions: {persona.previous_interactions}")
    
    print(f"\n✅ Persona Generator working correctly!")
