# --- Embedded CSS ---
css_string = """
<style>
    /* Basic Styling for demonstration */
    .intro-text {
        font-size: 1.1rem;
        color: #333;
        line-height: 1.6;
    }
    .prediction-container {
        text-align: center;
        padding: 20px;
        border: 1px solid #ddd;
        border-radius: 8px;
        background-color: #f9f9f9;
        min-height: 150px; /* Ensure consistent height */
        display: flex;
        flex-direction: column;
        justify-content: center;
    }
    .prediction-container h2 {
        font-size: 3rem;
        color: #007bff; /* Blue color for emphasis */
        margin-bottom: 5px;
    }
    .prediction-container p {
        font-size: 1rem;
        color: #555;
        margin-top: 0;
    }
    .stImage > img {
        max-width: 100%;
        height: auto;
        border: 1px solid #eee; /* Added for better visibility of the grid image */
        border-radius: 4px;   /* Added for softer corners */
    }
    .stButton>button { /* Ensure buttons in columns take full width if desired */
        width: 100%;
    }
</style>
"""