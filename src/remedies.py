"""
Agronomic disease management recommendations and diagnostic details
for potato crop diseases.
"""

DISEASE_INFO = {
    "Potato___Early_blight": {
        "common_name": "Early Blight",
        "causal_organism": "Alternaria solani (Fungal pathogen)",
        "symptoms": [
            "Concentric rings or 'target-board' necrotic brown spots on older leaves.",
            "Yellow chlorotic halos surrounding lesions.",
            "Premature defoliation starting from the lower canopy."
        ],
        "severity": "Moderate to High",
        "chemical_control": [
            "Apply protectant fungicides such as Mancozeb (75% WP) or Chlorothalonil early in the season.",
            "Rotate with systemic fungicides like Difenoconazole or Azoxystrobin upon first symptom detection.",
            "Ensure minimum 7–10 day spray intervals during humid weather."
        ],
        "cultural_practices": [
            "Practice 3–4 year crop rotation with non-solanaceous crops (e.g., maize, legumes).",
            "Use certified, disease-free seed tubers.",
            "Avoid overhead irrigation to minimize leaf wetness duration; use drip irrigation where possible.",
            "Destroy and bury infected crop residues immediately after harvest."
        ]
    },
    "Potato___Late_blight": {
        "common_name": "Late Blight",
        "causal_organism": "Phytophthora infestans (Oomycete pathogen)",
        "symptoms": [
            "Irregular, rapidly enlarging water-soaked pale-green to black lesions on leaves and stems.",
            "White velvety fungal growth (sporulation) visible on lesion undersides under high humidity.",
            "Entire vines blighted and rotten within days if untreated, emitting a characteristic pungent odor."
        ],
        "severity": "CRITICAL / EMERGENCY",
        "chemical_control": [
            "Immediate emergency application of curative fungicides: Metalaxyl-M + Mancozeb (Ridomil Gold).",
            "Follow up with Cymoxanil or Dimethomorph to halt mycelial spread.",
            "Ensure complete foliage coverage, especially the undersides of leaves."
        ],
        "cultural_practices": [
            "Scout fields daily during cool (10–20°C), foggy, or high-humidity weather.",
            "Destroy all volunteer potato plants and cull piles in the vicinity.",
            "Plant resistant or tolerant potato cultivars (e.g., Gudene, Belete in Ethiopian highland varieties).",
            "Halt all irrigation when blight is active to prevent zoospore dissemination."
        ]
    },
    "Potato___healthy": {
        "common_name": "Healthy Foliage",
        "causal_organism": "None (Plant is in optimal vigor)",
        "symptoms": [
            "Uniform green leaf color across the blade.",
            "No necrotic spots, chlorotic halos, or wilting observed.",
            "Turgid leaf structure and normal venation."
        ],
        "severity": "None (Optimal condition)",
        "chemical_control": [
            "No chemical intervention required.",
            "Optional preventive bio-fungicide (Trichoderma spp.) or mild copper spray if neighboring fields report disease."
        ],
        "cultural_practices": [
            "Maintain balanced N-P-K fertilization (avoid excessive nitrogen which promotes soft, vulnerable tissue).",
            "Maintain consistent soil moisture via regulated drip irrigation.",
            "Continue periodic preventive scouting twice a week."
        ]
    }
}
