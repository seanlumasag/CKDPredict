# Use OpenJDK 17 slim image
FROM openjdk:17-jdk-slim

# Copy the jar file to the container
COPY target/your-app-0.0.1-SNAPSHOT.jar app.jar

# Expose the port your app listens on (Render expects PORT env var, usually 10000)
EXPOSE 10000

# Run the app
ENTRYPOINT ["java", "-jar", "/app.jar"]
