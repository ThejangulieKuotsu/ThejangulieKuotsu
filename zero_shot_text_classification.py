# pip install gliner2
from gliner2 import GLiNER2

extractor = GLiNER2.from_pretrained("fastino/gliner2-large-v1")

text = "Patient received 400mg ibuprofen for severe headache at 2 PM."
result = extractor.extract_entities(
    text,
    {
        "medication": "Names of drugs, medications, or pharmaceutical substances",
        "dosage": "Specific amounts like '2000mg', '2 tablets', or '5ml'",
        "symptom": "Medical symptoms, conditions, or patient complaints",
        "time": "Time references like '2 PM', 'morning', or 'after lunch'"
    }
)

print(result)
