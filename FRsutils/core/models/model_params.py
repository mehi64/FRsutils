

model_parameters = {
    "fr_model": 
    {
        "ITFRS": 
        {
            "required": {"tnorm": {"type": str, "choices": {"tnorm1", "tnorm2", "tnorm3"}}},
            "optional": {"verbose": {"type": bool, "default": False}},
        },
        "OWAFRS": 
        {
            "required": {"tnorm": {"type": str, "choices": {"tnorm1", "tnorm2", "tnorm3"}}},
            "optional": {"verbose": {"type": bool, "default": False}},
        },
        "VQRS": 
        {
            "required": {
                "alpha_lower": {"type": float},
                "beta_lower": {"type": float}
            },
            "optional": {"gamma": {"type": float, "default": 0.5}},
        },
    },
    "hgf": {
        "HGF1": {
            "required": {"learning_rate": {"type": float}},
            "optional": {"momentum": {"type": float, "default": 0.9}},
        },
        "HGF2": {
            "required": {},
            "optional": {"decay": {"type": float, "default": 0.01}},
        },
    }
}