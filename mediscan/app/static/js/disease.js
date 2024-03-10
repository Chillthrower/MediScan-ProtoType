// var words = [
//     "itching", "skin_rash", "nodal_skin_eruptions", "continuous_sneezing", "shivering", "chills", "joint_pain", 
//     "stomach_pain", "acidity", "ulcers_on_tongue", "muscle_wasting", "vomiting", "burning_micturition", 
//     "spotting_ urination", "fatigue", "weight_gain", "anxiety", "cold_hands_and_feets", "mood_swings", 
//     "weight_loss", "restlessness", "lethargy", "patches_in_throat", "irregular_sugar_level", "cough", 
//     "high_fever", "sunken_eyes", "breathlessness", "sweating", "dehydration", "indigestion", "headache", 
//     "yellowish_skin", "dark_urine", "nausea", "loss_of_appetite", "pain_behind_the_eyes", "back_pain", 
//     "constipation", "abdominal_pain", "diarrhoea", "mild_fever", "yellow_urine", "yellowing_of_eyes", 
//     "acute_liver_failure", "fluid_overload", "swelling_of_stomach", "swelled_lymph_nodes", "malaise", 
//     "blurred_and_distorted_vision", "phlegm", "throat_irritation", "redness_of_eyes", "sinus_pressure", 
//     "runny_nose", "congestion", "chest_pain", "weakness_in_limbs", "fast_heart_rate", "pain_during_bowel_movements", 
//     "pain_in_anal_region", "bloody_stool", "irritation_in_anus", "neck_pain", "dizziness", "cramps", 
//     "bruising", "obesity", "swollen_legs", "swollen_blood_vessels", "puffy_face_and_eyes", "enlarged_thyroid", 
//     "brittle_nails", "swollen_extremeties", "excessive_hunger", "extra_marital_contacts", "drying_and_tingling_lips", 
//     "slurred_speech", "knee_pain", "hip_joint_pain", "muscle_weakness", "stiff_neck", "swelling_joints", 
//     "movement_stiffness", "spinning_movements", "loss_of_balance", "unsteadiness", "weakness_of_one_body_side", 
//     "loss_of_smell", "bladder_discomfort", "foul_smell_of urine", "continuous_feel_of_urine", "passage_of_gases", 
//     "internal_itching", "toxic_look_(typhos)", "depression", "irritability", "muscle_pain", "altered_sensorium", 
//     "red_spots_over_body", "belly_pain", "abnormal_menstruation", "dischromic _patches", "watering_from_eyes", 
//     "increased_appetite", "polyuria", "family_history", "mucoid_sputum", "rusty_sputum", "lack_of_concentration", 
//     "visual_disturbances", "receiving_blood_transfusion", "receiving_unsterile_injections", "coma", 
//     "stomach_bleeding", "distention_of_abdomen", "history_of_alcohol_consumption", "fluid_overload", 
//     "blood_in_sputum", "prominent_veins_on_calf", "palpitations", "painful_walking", "pus_filled_pimples", 
//     "blackheads", "scurring", "skin_peeling", "silver_like_dusting", "small_dents_in_nails", "inflammatory_nails", 
//     "blister", "red_sore_around_nose", "yellow_crust_ooze", "prognosis"
// ];

$(document).ready(function() {
    $('#symptomForm').submit(function(event) {
        event.preventDefault();
        
        var symptoms = $('#symptoms').val();
        if (symptoms) {
            predictDisease(symptoms);
        }
    });
});

function createTextboxes() {
    var numTextboxes = parseInt($('#numTextboxes').val());
    var textboxContainer = $('#textboxContainer');
    textboxContainer.empty();
    for (var i = 0; i < numTextboxes; i++) {
        var textbox = $('<input>').attr({
            type: 'text',
            name: 'symptoms[]', 
            placeholder: 'Enter symptom ' + (i + 1)
        });
        textboxContainer.append(textbox);
    }
}

function submitForm() {
    var symptoms = $('input[name="symptoms[]"]').map(function(){return $(this).val();}).get();
    predictDisease(symptoms);
}

function predictDisease(symptoms) {
    $.ajax({
        type: 'POST',
        url: '/predict_disease/',
        data: {
            'symptoms': symptoms
        },
        success: function(response) {
            displayPredictions(response);
        },
        error: function(xhr, errmsg, err) {
            console.log(xhr.status + ": " + xhr.responseText);
        }
    });
}

function displayPredictions(predictions) {
    $('#predictionsContainer').html(`
        <h2>Predictions</h2>
        <p>Random Forest Model Prediction: ${predictions.rf_model_prediction}</p>
        <p>Naive Bayes Model Prediction: ${predictions.naive_bayes_prediction}</p>
        <p>SVM Model Prediction: ${predictions.svm_model_prediction}</p>
        <p>Final Prediction: ${predictions.final_prediction}</p>
        <p>Information: ${predictions.treatment}</p>
    `);
}