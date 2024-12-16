PRICING = {
    None: {
        "currency": "$",
        "unit": 1_000,
        "step": {},
        "default":{
            "prompt": 0.0,
            "completion": 0.0,
        }
    },
    "gemini-1.5-flash": {
        "currency": "$",
        "unit": 1_000_000,
        "step": {
            128_000:{
                "prompt": 0.15,
                "completion": 0.6,
            }
        },
        "default":{
            "prompt": 0.0375,
            "completion": 0.3,
        }
    },
    "gemini-1.5-pro": {
        "currency": "$",
        "unit": 1_000_000,
        "step": {
            128_000:{
                "prompt": 2.5,
                "completion": 10.00,
            }
        },
        "default":{
            "prompt": 1.25,
            "completion": 5.0,
        }
    },
    "gpt-4o": {
        "currency": "$",
        "unit": 1_000_000,
        "step": {},
        "default": {
            "prompt": 2.5,
            "completion": 10.0
        }
    },
    "gpt-4o-2024-11-20": {
        "currency": "$",
        "unit": 1_000_000,
        "step": {},
        "default": {
            "prompt": 2.5,
            "completion": 10.0
        }
    },
    "gpt-4o-2024-08-06": {
        "currency": "$",
        "unit": 1_000_000,
        "step": {},
        "default": {
            "prompt": 2.5,
            "completion": 10.0
        }
    },
    "gpt-4o-audio-preview": {
        "currency": "$",
        "unit": 1_000_000,
        "step": {},
        "default": {
            "prompt_text": 2.5,
            "completion_text": 10.0,
            "prompt_audio": 100.0,
            "completion_audio": 200.0
        }
    },
    "gpt-4o-audio-preview-2024-10-01": {
        "currency": "$",
        "unit": 1_000_000,
        "step": {},
        "default": {
            "prompt_text": 2.5,
            "completion_text": 10.0,
            "prompt_audio": 100.0,
            "completion_audio": 200.0
        }
    },
    "gpt-4o-2024-05-13": {
        "currency": "$",
        "unit": 1_000_000,
        "step": {},
        "default": {
            "prompt": 5.0,
            "completion": 15.0
        }
    },
    "gpt-4o-mini": {
        "currency": "$",
        "unit": 1_000_000,
        "step": {},
        "default": {
            "prompt": 0.15,
            "completion": 0.6
        }
    },
    "gpt-4o-mini-2024-07-18": {
        "currency": "$",
        "unit": 1_000_000,
        "step": {},
        "default": {
            "prompt": 0.15,
            "completion": 0.6
        }
    },
    "o1-preview": {
        "currency": "$",
        "unit": 1_000_000,
        "step": {},
        "default": {
            "prompt": 15.0,
            "completion": 60.0
        }
    },
    "o1-preview-2024-09-12": {
        "currency": "$",
        "unit": 1_000_000,
        "step": {},
        "default": {
            "prompt": 15.0,
            "completion": 60.0
        }
    },
    "o1-mini": {
        "currency": "$",
        "unit": 1_000_000,
        "step": {},
        "default": {
            "prompt": 3.0,
            "completion": 12.0
        }
    },
    "o1-mini-2024-09-12": {
        "currency": "$",
        "unit": 1_000_000,
        "step": {},
        "default": {
            "prompt": 3.0,
            "completion": 12.0
        }
    }
}
