test_cases = [
    {
        "predictions": ("A", "B", "C"),
        "references": ("A", "C", "B"),
        "result": {
            "kendall_tau_distance": 1.0,
            "normalized_kendall_tau_distance": 0.3333333333333333,
        },
    },
    {
        "predictions": ("A", "B", "C"),
        "references": ("B", "A", "C"),
        "result": {
            "kendall_tau_distance": 1.0,
            "normalized_kendall_tau_distance": 0.3333333333333333,
        },
    },
    {
        "predictions": ("A", "B", "C"),
        "references": ("B", "C", "A"),
        "result": {
            "kendall_tau_distance": 2.0,
            "normalized_kendall_tau_distance": 0.6666666666666666,
        },
    },
    {
        "predictions": ("A", "B", "C"),
        "references": ("C", "A", "B"),
        "result": {
            "kendall_tau_distance": 2.0,
            "normalized_kendall_tau_distance": 0.6666666666666666,
        },
    },
    {
        "predictions": ("A", "B", "C"),
        "references": ("C", "B", "A"),
        "result": {"kendall_tau_distance": 3.0, "normalized_kendall_tau_distance": 1.0},
    },
    {
        "predictions": ("A", "C", "B"),
        "references": ("B", "A", "C"),
        "result": {
            "kendall_tau_distance": 2.0,
            "normalized_kendall_tau_distance": 0.6666666666666666,
        },
    },
    {
        "predictions": ("A", "C", "B"),
        "references": ("B", "C", "A"),
        "result": {"kendall_tau_distance": 3.0, "normalized_kendall_tau_distance": 1.0},
    },
    {
        "predictions": ("A", "C", "B"),
        "references": ("C", "A", "B"),
        "result": {
            "kendall_tau_distance": 1.0,
            "normalized_kendall_tau_distance": 0.3333333333333333,
        },
    },
    {
        "predictions": ("A", "C", "B"),
        "references": ("C", "B", "A"),
        "result": {
            "kendall_tau_distance": 2.0,
            "normalized_kendall_tau_distance": 0.6666666666666666,
        },
    },
    {
        "predictions": ("B", "A", "C"),
        "references": ("B", "C", "A"),
        "result": {
            "kendall_tau_distance": 1.0,
            "normalized_kendall_tau_distance": 0.3333333333333333,
        },
    },
    {
        "predictions": ("B", "A", "C"),
        "references": ("C", "A", "B"),
        "result": {"kendall_tau_distance": 3.0, "normalized_kendall_tau_distance": 1.0},
    },
    {
        "predictions": ("B", "A", "C"),
        "references": ("C", "B", "A"),
        "result": {
            "kendall_tau_distance": 2.0,
            "normalized_kendall_tau_distance": 0.6666666666666666,
        },
    },
    {
        "predictions": ("B", "C", "A"),
        "references": ("C", "A", "B"),
        "result": {
            "kendall_tau_distance": 2.0,
            "normalized_kendall_tau_distance": 0.6666666666666666,
        },
    },
    {
        "predictions": ("B", "C", "A"),
        "references": ("C", "B", "A"),
        "result": {
            "kendall_tau_distance": 1.0,
            "normalized_kendall_tau_distance": 0.3333333333333333,
        },
    },
    {
        "predictions": ("C", "A", "B"),
        "references": ("C", "B", "A"),
        "result": {
            "kendall_tau_distance": 1.0,
            "normalized_kendall_tau_distance": 0.3333333333333333,
        },
    },
]
