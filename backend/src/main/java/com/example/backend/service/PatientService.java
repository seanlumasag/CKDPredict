package com.example.backend.service;

import com.example.backend.model.Patient;
import com.example.backend.repository.PatientRepository;
import org.springframework.stereotype.Service;

import java.util.List;
import java.util.Optional;

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
        boolean predictedRisk = dummyPredict(patient);
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
        boolean predictedRisk = dummyPredict(existingPatient);
        existingPatient.setPrediction(predictedRisk);

        return patientRepository.save(existingPatient);
    }

    // Dummy prediction logic for example — replace with real ML call/integration
    private boolean dummyPredict(Patient patient) {
        // Example: if blood pressure > 140, high risk
        return patient.getBp() > 140;
    }
}
