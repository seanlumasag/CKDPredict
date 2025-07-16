package com.example.backend.service;

import com.example.backend.model.Patient;
import com.example.backend.repository.PatientRepository;
import org.springframework.stereotype.Service;

import java.util.List;
import java.util.Optional;

import org.springframework.web.client.RestTemplate;
import org.springframework.http.ResponseEntity;
import org.springframework.core.ParameterizedTypeReference;
import org.springframework.http.HttpEntity;
import org.springframework.http.HttpHeaders;
import org.springframework.http.MediaType;
import java.util.HashMap;
import java.util.Map;

@Service
public class PatientService {

    private final PatientRepository patientRepository;

    // Inject model/scaler dependencies here if you have them
    // private final MLModel mlModel;

    public PatientService(PatientRepository patientRepository) {
        this.patientRepository = patientRepository;
        // this.mlModel = mlModel; // if applicable
    }

    public List<Patient> getAllPatients() {
        return patientRepository.findAll();
    }

    public Optional<Patient> getPatientById(Long id) {
        return patientRepository.findById(id);
    }

    public Patient savePatient(Patient patient) {
        return patientRepository.save(patient);
    }

    public void deletePatient(Long id) {
        patientRepository.deleteById(id);
    }

    public Patient predictAndSave(Patient patient) {
        // TODO: Add actual ML prediction logic here,
        // For now, dummy example:
        boolean predictedRisk = MLPredict(patient);
        patient.setPrediction(predictedRisk);

        return patientRepository.save(patient);
    }

    public List<Patient> getRecentPatients() {
        // Assuming you add this method in PatientRepository:
        return patientRepository.findTop10ByOrderByCreatedAtDesc();
    }

    public Patient updatePatient(Long id, Patient updatedPatient) {
        Patient existingPatient = patientRepository.findById(id)
                .orElseThrow(() -> new RuntimeException("Patient not found"));

        existingPatient.setAge(updatedPatient.getAge());
        existingPatient.setBp(updatedPatient.getBp());
        existingPatient.setSg(updatedPatient.getSg());
        existingPatient.setAl(updatedPatient.getAl());

        // Recalculate prediction (dummy for now)
        boolean predictedRisk = MLPredict(existingPatient);
        existingPatient.setPrediction(predictedRisk);

        return patientRepository.save(existingPatient);
    }

    private boolean MLPredict(Patient patient) {
        try {
            RestTemplate restTemplate = new RestTemplate();

            String mlUrl_ = "http://localhost:8000/predict";
            String mlUrl = "https://backend-ml-3d3w.onrender.com/predict";

            // Prepare request body as Map
            Map<String, Object> requestBody = new HashMap<>();
            requestBody.put("age", patient.getAge());
            requestBody.put("bp", patient.getBp());
            requestBody.put("sg", patient.getSg());
            requestBody.put("al", patient.getAl());

            // Set headers
            HttpHeaders headers = new HttpHeaders();
            headers.setContentType(MediaType.APPLICATION_JSON);

            HttpEntity<Map<String, Object>> request = new HttpEntity<>(requestBody, headers);

            // Call FastAPI POST /predict endpoint
            ResponseEntity<Map<String, Object>> response = restTemplate.exchange(
                    mlUrl,
                    org.springframework.http.HttpMethod.POST,
                    request,
                    new ParameterizedTypeReference<Map<String, Object>>() {
                    });

            if (response.getStatusCode().is2xxSuccessful() && response.getBody() != null) {
                Map<String, Object> body = response.getBody();
                // Assuming your FastAPI response has a key "prediction" with 0 or 1
                Integer prediction = (Integer) body.get("prediction");
                return prediction != null && prediction == 1;
            }
        } catch (Exception e) {
            e.printStackTrace();
            // Log error or handle accordingly
        }

        // Fallback to false if call fails
        return false;
    }
}
