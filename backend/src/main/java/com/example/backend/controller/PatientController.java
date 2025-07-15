package com.example.backend.controller;

import com.example.backend.model.Patient;
import com.example.backend.service.PatientService;
import org.springframework.beans.factory.annotation.Autowired;
import org.springframework.http.ResponseEntity;
import org.springframework.web.bind.annotation.*;

import java.util.List;

@RestController
@RequestMapping("/api")
public class PatientController {

    @Autowired
    private PatientService patientService;

    @PostMapping("/predict")
    public ResponseEntity<?> createPrediction(@RequestBody Patient patient) {
        // Save patient and return prediction result
        return ResponseEntity.ok(patientService.predictAndSave(patient));
    }

    @GetMapping("/recent")
    public ResponseEntity<List<Patient>> getRecentPatients() {
        return ResponseEntity.ok(patientService.getRecentPatients());
    }

    @PutMapping("/patient/{id}")
    public ResponseEntity<?> updatePatient(@PathVariable Long id, @RequestBody Patient patient) {
        return ResponseEntity.ok(patientService.updatePatient(id, patient));
    }

    @DeleteMapping("/patient/{id}")
    public ResponseEntity<Void> deletePatient(@PathVariable Long id) {
        patientService.deletePatient(id);
        return ResponseEntity.noContent().build();
    }
}
