const form = document.getElementById("predictionForm");

const predictBtn = document.getElementById("predictBtn");

const emptyState = document.getElementById("emptyState");

const resultContent =
    document.getElementById("resultContent");

const resultTitle =
    document.getElementById("resultTitle");

const resultIcon =
    document.getElementById("resultIcon");

const probability =
    document.getElementById("probability");

const progressBar =
    document.getElementById("progressBar");

const resultStatus =
    document.getElementById("resultStatus");

const resultPanel =
    document.querySelector(".result-panel");

const errorBox =
    document.getElementById("errorBox");

const resetBtn =
    document.getElementById("resetBtn");

const explanationBox = document.createElement("div");
explanationBox.className = "explanation-box";
resultContent.appendChild(explanationBox);

/* ============================
   PREDICTION
============================ */

form.addEventListener("submit", async function(event) {

    event.preventDefault();


    // Hide old messages
    errorBox.style.display = "none";

    resultContent.style.display = "none";


    // Button loading state
    predictBtn.disabled = true;

    predictBtn.innerHTML =
        "Analyzing Patient <span class='arrow'>...</span>";


    // Collect patient data
    const patientData = {

        age: parseFloat(
            document.getElementById("age").value
        ),

        sex: parseInt(
            document.getElementById("sex").value
        ),

        chest_pain_type: parseInt(
            document.getElementById("chest_pain_type").value
        ),

        resting_bp_s: parseFloat(
            document.getElementById("resting_bp_s").value
        ),

        cholesterol: parseFloat(
            document.getElementById("cholesterol").value
        ),

        fasting_blood_sugar: parseInt(
            document.getElementById("fasting_blood_sugar").value
        ),

        resting_ecg: parseInt(
            document.getElementById("resting_ecg").value
        ),

        max_heart_rate: parseFloat(
            document.getElementById("max_heart_rate").value
        ),

        exercise_angina: parseInt(
            document.getElementById("exercise_angina").value
        ),

        oldpeak: parseFloat(
            document.getElementById("oldpeak").value
        ),

        ST_slope: parseInt(
            document.getElementById("ST_slope").value
        )
    };


    try {

        const response = await fetch(
            "/predict",
            {
                method: "POST",

                headers: {
                    "Content-Type": "application/json"
                },

                body: JSON.stringify(patientData)
            }
        );


        const data = await response.json();


        if (!response.ok) {

            let errorMessage = "Prediction failed. Please check your inputs.";

            if (response.status === 422) {

                if (Array.isArray(data.detail)) {

                    errorMessage =
                        "Please check the following information:\n\n" +
                        data.detail
                            .map(error => {
                                const field =
                                    error.loc[error.loc.length - 1];

                                return `• ${field}: ${error.msg}`;
                            })
                            .join("\n");

                } else {

                    errorMessage =
                        "Some patient information is invalid.";

                }

            } else if (data.detail) {

                errorMessage = data.detail;

            }

            throw new Error(errorMessage);
        }


        /* ============================
           SHOW RESULT
        ============================ */

        emptyState.style.display = "none";

        resultContent.style.display = "block";


        const probabilityValue =
            Number(data.heart_disease_probability) * 100;


        probability.textContent =
            probabilityValue.toFixed(2) + "%";


        resultTitle.textContent =
            data.result;


        progressBar.style.width =
            probabilityValue + "%";


        resultPanel.classList.remove(
            "positive",
            "negative"
        );


        /* HEART DISEASE */

        if (data.prediction === 1) {

            resultPanel.classList.add("positive");

            resultIcon.textContent = "!";
            
            resultStatus.textContent =
                "Heart Disease";

        }


        /* NO HEART DISEASE */

        else {

            resultPanel.classList.add("negative");

            resultIcon.textContent = "✓";

            resultStatus.textContent =
                "No Heart Disease";

        }

        /* ============================
        SHAP EXPLANATIONS
        ============================ */

        explanationBox.innerHTML = "";

        const explanationHeading =
            document.createElement("h3");

        explanationHeading.textContent =
            "Why did the model make this prediction?";

        explanationBox.appendChild(
            explanationHeading
        );


        data.explanations.forEach(item => {

            const explanationItem =
                document.createElement("div");

            explanationItem.className =
                item.contribution > 0
                    ? "explanation-item risk"
                    : "explanation-item protective";


            const icon =
                item.contribution > 0
                    ? "↑"
                    : "↓";


            explanationItem.innerHTML = `
                <div class="explanation-icon">
                    ${icon}
                </div>

                <div class="explanation-text">
                    <strong>${item.feature}</strong>
                    <span>${item.direction}</span>
                </div>
            `;


            explanationBox.appendChild(
                explanationItem
            );

        });

    }


    catch (error) {

        errorBox.textContent = error.message;

        errorBox.style.display = "block";

        // Scroll to the error message
        errorBox.scrollIntoView({
            behavior: "smooth",
            block: "center"
        });

    }


    finally {

        predictBtn.disabled = false;

        predictBtn.innerHTML =
            "<span class='button-icon'>✦</span>" +
            " Analyze Patient " +
            "<span class='arrow'>→</span>";

    }

});


/* ============================
   RESET
============================ */

resetBtn.addEventListener(
    "click",
    function() {

        form.reset();

        emptyState.style.display =
            "flex";

        resultContent.style.display =
            "none";

        errorBox.style.display =
            "none";
        
        explanationBox.innerHTML = "";
        
        progressBar.style.width =
            "0%";

        resultPanel.classList.remove(
            "positive",
            "negative"
        );

        window.scrollTo({
            top: 0,
            behavior: "smooth"
        });

    }
);