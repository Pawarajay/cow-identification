document.addEventListener('DOMContentLoaded', () => {
    // --- Cow Identification Form Handling ---
    const cowIdForm = document.getElementById('cowIdForm');
    const cowIdResultDiv = document.getElementById('cowIdResult');
    const uploadedCowImageContainer = document.getElementById('uploadedCowImageContainer');

    cowIdForm.addEventListener('submit', async (e) => {
        e.preventDefault();
        cowIdResultDiv.innerHTML = 'Identifying cow...';
        uploadedCowImageContainer.innerHTML = '';

        const formData = new FormData(cowIdForm);
        const cowImageInput = document.getElementById('cowImage');

        if (cowImageInput.files.length === 0) {
            cowIdResultDiv.innerHTML = '<p class="error">Please select an image file for identification.</p>';
            return;
        }

        try {
            const response = await fetch('/identify_cow', {
                method: 'POST',
                body: formData
            });
            const data = await response.json();

            if (data.success) {
                cowIdResultDiv.innerHTML = `
                    <p><strong>Identified Cow ID:</strong> ${data.cow_id}</p>
                    <p><strong>Confidence:</strong> ${data.confidence}</p>
                `;
                if (data.image_url) {
                    uploadedCowImageContainer.innerHTML = `
                        <h3>Uploaded Cow Image:</h3>
                        <img src="${data.image_url}" alt="Uploaded Cow Image">
                    `;
                }
            } else {
                cowIdResultDiv.innerHTML = `<p class="error">Error: ${data.error}</p>`;
            }
        } catch (error) {
            console.error('Error during cow identification:', error);
            cowIdResultDiv.innerHTML = `<p class="error">An unexpected error occurred. Please check the console.</p>`;
        }
    });

    // --- Health Prediction Form Handling (Conjunctivitis) ---
    const healthForm = document.getElementById('healthForm');
    const healthResultDiv = document.getElementById('healthResult');
    const uploadedEyeImageContainer = document.getElementById('uploadedEyeImageContainer');

    healthForm.addEventListener('submit', async (e) => {
        e.preventDefault();
        healthResultDiv.innerHTML = 'Predicting health status...';
        uploadedEyeImageContainer.innerHTML = '';

        const formData = new FormData(healthForm);
        const eyeImageInput = document.getElementById('eyeImage');

        if (eyeImageInput.files.length === 0) {
            healthResultDiv.innerHTML = '<p class="error">Please select an eye image file for health prediction.</p>';
            return;
        }

        try {
            const response = await fetch('/predict_health', {
                method: 'POST',
                body: formData
            });
            const data = await response.json();

            if (data.success) {
                healthResultDiv.innerHTML = `
                    <p><strong>Health Status:</strong> <span class="${data.status.toLowerCase().replace(' ', '-')}">${data.status}</span></p>
                    <p>Prediction Score: ${data.prediction_score}</p>
                `;
                if (data.image_url) {
                    uploadedEyeImageContainer.innerHTML = `
                        <h3>Uploaded Eye Image:</h3>
                        <img src="${data.image_url}" alt="Uploaded Eye Image">
                    `;
                }
            } else {
                healthResultDiv.innerHTML = `<p class="error">Error: ${data.error}</p>`;
            }
        } catch (error) {
            console.error('Error during health prediction:', error);
            healthResultDiv.innerHTML = `<p class="error">An unexpected error occurred. Please check the console.</p>`;
        }
    });
});