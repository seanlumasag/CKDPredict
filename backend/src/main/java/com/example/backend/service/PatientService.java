package com.example.backend.service;

import com.example.backend.model.Patient;
import com.example.backend.repository.PatientRepository;
import org.springframework.stereotype.Service;

import java.util.List;
import java.util.Optional;

import org.springframework.web.client.RestTemplate;
import org.springframework.http.ResponseEntity;
import org.springframework.beans.factory.annotation.Value;
import org.springframework.core.ParameterizedTypeReference;
import org.springframework.http.HttpEntity;
import org.springframework.http.HttpHeaders;
import org.springframework.http.MediaType;
import java.util.HashMap;
import java.util.Map;

/**
 * Service class that contains business logic related to Patient operations.
 * This includes database interactions and communication with the external ML
 * prediction service.
 */
@Service
public class PatientService {

    private final PatientRepository patientRepository;

    // Constructor injection of the repository
    public PatientService(PatientRepository patientRepository) {
        this.patientRepository = patientRepository;
    }

    /**
     * @return All patients in the database.
     */
    public List<Patient> getAllPatients() {
        return patientRepository.findAll();
    }

    /**
     * Finds a patient by ID.
     * 
     * @param id The ID of the patient.
     * @return Optional containing the patient if found, otherwise empty.
     */
    public Optional<Patient> getPatientById(Long id) {
        return patientRepository.findById(id);
    }

    /**
     * Saves a new or existing patient to the database.
     * 
     * @param patient The patient to save.
     * @return The saved patient object.
     */
    public Patient savePatient(Patient patient) {
        return patientRepository.save(patient);
    }

    /**
     * Deletes a patient by their ID.
     * 
     * @param id The ID of the patient to delete.
     */
    public void deletePatient(Long id) {
        patientRepository.deleteById(id);
    }

    /**
     * Predicts CKD risk using the ML model, then saves the patient with the
     * prediction result.
     * 
     * @param patient The patient data to predict and save.
     * @return The saved patient object with prediction result.
     */
    public Patient predictAndSave(Patient patient) {
        float predictedRisk = MLPredict(patient);
        patient.setPrediction(predictedRisk);
        return patientRepository.save(patient);
    }

    /**
     * Returns the 10 most recently created patients.
     */
    public List<Patient> getRecentPatients() {
        return patientRepository.findTop10ByOrderByCreatedAtDesc();
    }

    /**
     * Updates an existing patient with new values and re-runs prediction.
     * 
     * @param id             The ID of the patient to update.
     * @param updatedPatient The updated patient data.
     * @return The updated and saved patient.
     */
    public Patient updatePatient(Long id, Patient updatedPatient) {
        // Fetch existing patient or throw if not found
        Patient existingPatient = patientRepository.findById(id)
                .orElseThrow(() -> new RuntimeException("Patient not found"));

        // Update fields
        existingPatient.setAge(updatedPatient.getAge());
        existingPatient.setBp(updatedPatient.getBp());
        existingPatient.setSg(updatedPatient.getSg());
        existingPatient.setAl(updatedPatient.getAl());

        // Predict again with updated data
        float predictedRisk = MLPredict(existingPatient);
        existingPatient.setPrediction(predictedRisk);

        // Save updated patient
        return patientRepository.save(existingPatient);
    }

    // URL of the external ML model service, injected from application.properties.
    @Value("${ml.service.url}")
    private String mlServiceUrl;

    /**
     * Calls the external ML service to predict CKD risk for a patient.
     * Sends a POST request with patient's age, bp, sg, al values.
     * Expects a JSON response: { "prediction": 0, 1, etc }.
     * 
     * @param patient The patient to send for prediction.
     * @return true if CKD is predicted (1), false otherwise (0 or error).
     */
    private float MLPredict(Patient patient) {

        try {
            RestTemplate restTemplate = new RestTemplate();

            // Full ML endpoint URL
            String mlUrl = mlServiceUrl + "/predict";

            // Prepare request body
            Map<String, Object> requestBody = new HashMap<>();
            requestBody.put("age", patient.getAge());
            requestBody.put("bp", patient.getBp());
            requestBody.put("sg", patient.getSg());
            requestBody.put("al", patient.getAl());

            // Set headers
            HttpHeaders headers = new HttpHeaders();
            headers.setContentType(MediaType.APPLICATION_JSON);

            // Build full HTTP request
            HttpEntity<Map<String, Object>> request = new HttpEntity<>(requestBody, headers);

            // Send POST request to ML service
            ResponseEntity<Map<String, Object>> response = restTemplate.exchange(
                    mlUrl,
                    org.springframework.http.HttpMethod.POST,
                    request,
                    new ParameterizedTypeReference<Map<String, Object>>() {
                    });

            // Check response and extract prediction
            if (response.getStatusCode().is2xxSuccessful() && response.getBody() != null) {
                Map<String, Object> body = response.getBody();
                float prediction = ((Number) body.get("prediction")).floatValue();
                return prediction;
            }
        } catch (Exception e) {
            // Log error and return false
            e.printStackTrace();
        }

        return -1;
    }
}
