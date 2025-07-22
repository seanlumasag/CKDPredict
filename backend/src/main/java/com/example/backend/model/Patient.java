package com.example.backend.model;

import jakarta.persistence.*;
import java.time.LocalDateTime;
import java.util.UUID;

/**
 * Entity representing a patient and their CKD prediction data.
 * This maps to the 'patient' table in the database.
 */
@Entity
@Table(name = "patient")
public class Patient {

    /**
     * Primary key for the patient record.
     * Value is auto-generated using UUID.
     */
    @Id
    @GeneratedValue
    private UUID id;

    // Age of patient
    private float age;

    // Blood pressure measurement
    private float bp;

    // Specific gravity of urine
    private float sg;

    // Albumin level
    private float al;

    // Model's prediction: true if CKD is likely, false otherwise
    private float prediction;

    /**
     * Timestamp when the record was created.
     * Automatically set before persisting, not updatable afterward.
     */
    @Column(name = "created_at", updatable = false)
    private LocalDateTime createdAt;

    /**
     * Sets the 'createdAt' timestamp to the current time
     * before the entity is persisted to the database.
     */
    @PrePersist
    protected void onCreate() {
        this.createdAt = LocalDateTime.now();
    }

    // Getters and setters
    public UUID getId() {
        return id;
    }

    public void setId(UUID id) {
        this.id = id;
    }

    public float getAge() {
        return age;
    }

    public void setAge(float age) {
        this.age = age;
    }

    public float getBp() {
        return bp;
    }

    public void setBp(float bp) {
        this.bp = bp;
    }

    public float getSg() {
        return sg;
    }

    public void setSg(float sg) {
        this.sg = sg;
    }

    public float getAl() {
        return al;
    }

    public void setAl(float al) {
        this.al = al;
    }

    public float getPrediction() {
        return prediction;
    }

    public void setPrediction(float prediction) {
        this.prediction = prediction;
    }

    public LocalDateTime getCreatedAt() {
        return createdAt;
    }
}