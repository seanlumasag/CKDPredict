package com.example.backend.controller;

import com.example.backend.model.Patient;
import com.example.backend.service.PatientService;
import org.springframework.beans.factory.annotation.Autowired;
import org.springframework.http.ResponseEntity;
import org.springframework.web.bind.annotation.*;

import java.util.List;

/**
 * REST controller handling Patient-related API endpoints.
 * 
 * Base path: /api
 * 
 * Exposes endpoints for:
 * - Creating CKD predictions
 * - Retrieving recent patient records
 * - Updating existing patient data
 * - Deleting patient records
 */
@RestController
@RequestMapping("/api")
public class PatientController {

    // Service layer dependency to handle business logic
    @Autowired
    private PatientService patientService;

    /**
     * POST /api/predict
     * Receives patient data, runs CKD prediction, saves the record, and returns the
     * result.
     * 
     * @param patient Patient object parsed from JSON request body
     * @return Prediction result along with saved patient info
     */
    @PostMapping("/predict")
    public ResponseEntity<?> createPrediction(@RequestBody Patient patient) {
        return ResponseEntity.ok(patientService.predictAndSave(patient));
    }

    /**
     * GET /api/recent
     * Retrieves a list of recent patient records.
     * 
     * @return List of Patient objects
     */
    @GetMapping("/recent")
    public ResponseEntity<List<Patient>> getRecentPatients() {
        return ResponseEntity.ok(patientService.getRecentPatients());
    }

    /**
     * PUT /api/patient/{id}
     * Updates patient data for the patient identified by the given ID.
     * 
     * @param id      Patient ID extracted from URL path
     * @param patient Patient data from JSON request body to update
     * @return Updated patient info or status
     */
    @PutMapping("/patient/{id}")
    public ResponseEntity<?> updatePatient(@PathVariable Long id, @RequestBody Patient patient) {
        return ResponseEntity.ok(patientService.updatePatient(id, patient));
    }

    /**
     * DELETE /api/patient/{id}
     * Deletes the patient record identified by the given ID.
     * 
     * @param id Patient ID extracted from URL path
     * @return HTTP 204 No Content on successful deletion
     */
    @DeleteMapping("/patient/{id}")
    public ResponseEntity<Void> deletePatient(@PathVariable Long id) {
        patientService.deletePatient(id);
        return ResponseEntity.noContent().build();
    }
}
