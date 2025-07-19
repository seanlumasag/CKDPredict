package com.example.backend.config;

import org.springframework.beans.factory.annotation.Value;
import org.springframework.context.annotation.Bean;
import org.springframework.context.annotation.Configuration;
import org.springframework.web.servlet.config.annotation.CorsRegistry;
import org.springframework.web.servlet.config.annotation.WebMvcConfigurer;

@Configuration
public class CorsConfig {

    // Inject the ML service URL from application.properties
    @Value("${ml.service.url}")
    private String mlServiceUrl;

    /**
     * Configure CORS settings for the backend.
     * This allows specified frontends and ml-services to make cross-origin
     * requests to the API.
     */
    @Bean
    public WebMvcConfigurer corsConfigurer() {
        return new WebMvcConfigurer() {
            @Override
            public void addCorsMappings(CorsRegistry registry) {
                registry.addMapping("/**") // Apply CORS rules to all endpoints
                        .allowedOrigins(

                                // Local development frontend (React)
                                "http://localhost:5173",

                                // Production frontend hosted on Vercel
                                "https://ckdpredict.vercel.app",

                                // Local development ML-microservice
                                "http://localhost:8000",

                                // Production ML-service hosted on Vercel
                                mlServiceUrl)
                        // Allow typical CRUD operations
                        .allowedMethods("GET", "POST", "PUT", "DELETE")

                        // Allow all headers (like Content-Type, Authorization)
                        .allowedHeaders("*");
            }
        };
    }
}
