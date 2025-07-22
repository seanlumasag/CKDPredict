package com.example.backend.repository;

import com.example.backend.model.Patient;
import org.springframework.data.jpa.repository.JpaRepository;
import org.springframework.stereotype.Repository;

import java.util.List;
import java.util.UUID;

/**
 * Repository interface for managing Patient entities.
 * Extends JpaRepository to provide CRUD operations and custom query methods.
 */
@Repository
public interface PatientRepository extends JpaRepository<Patient, UUID> {
    /**
     * Retrieves the 10 most recently created Patient records.
     * 
     * @return List of top 10 patients ordered by creation time (newest first)
     */
    List<Patient> findTop10ByOrderByCreatedAtDesc();
}
