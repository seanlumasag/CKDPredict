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

@Service
public class PatientService {

    private final PatientRepository patientRepository;

    public PatientService(PatientRepository patientRepository) {
        this.patientRepository = patientRepository;
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
        boolean predictedRisk = MLPredict(patient);
        patient.setPrediction(predictedRisk);

        return patientRepository.save(patient);
    }

    public List<Patient> getRecentPatients() {
        return patientRepository.findTop10ByOrderByCreatedAtDesc();
    }

    public Patient updatePatient(Long id, Patient updatedPatient) {
        Patient existingPatient = patientRepository.findById(id)
                .orElseThrow(() -> new RuntimeException("Patient not found"));

        existingPatient.setAge(updatedPatient.getAge());
        existingPatient.setBp(updatedPatient.getBp());
        existingPatient.setSg(updatedPatient.getSg());
        existingPatient.setAl(updatedPatient.getAl());

        boolean predictedRisk = MLPredict(existingPatient);
        existingPatient.setPrediction(predictedRisk);

        return patientRepository.save(existingPatient);
    }

    @Value("${ml.service.url}")
    private String mlServiceUrl;

    private boolean MLPredict(Patient patient) {

        try {
            RestTemplate restTemplate = new RestTemplate();

            String mlUrl = mlServiceUrl + "/predict";

            Map<String, Object> requestBody = new HashMap<>();
            requestBody.put("age", patient.getAge());
            requestBody.put("bp", patient.getBp());
            requestBody.put("sg", patient.getSg());
            requestBody.put("al", patient.getAl());

            HttpHeaders headers = new HttpHeaders();
            headers.setContentType(MediaType.APPLICATION_JSON);

            HttpEntity<Map<String, Object>> request = new HttpEntity<>(requestBody, headers);

            ResponseEntity<Map<String, Object>> response = restTemplate.exchange(
                    mlUrl,
                    org.springframework.http.HttpMethod.POST,
                    request,
                    new ParameterizedTypeReference<Map<String, Object>>() {
                    });

            if (response.getStatusCode().is2xxSuccessful() && response.getBody() != null) {
                Map<String, Object> body = response.getBody();
                Integer prediction = (Integer) body.get("prediction");
                return prediction != null && prediction == 1;
            }
        } catch (Exception e) {
            e.printStackTrace();
        }

        return false;
    }
}
