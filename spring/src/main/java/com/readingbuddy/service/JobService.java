package com.readingbuddy.service;

import com.readingbuddy.entity.Document;
import com.readingbuddy.entity.Job;
import com.readingbuddy.entity.JobStatus;
import com.readingbuddy.entity.User;
import com.readingbuddy.pipeline.PipelineRunner;
import com.readingbuddy.repository.JobRepository;
import org.springframework.beans.factory.annotation.Autowired;
import org.springframework.beans.factory.annotation.Value;
import org.springframework.data.domain.PageRequest;
import org.springframework.scheduling.annotation.Async;
import org.springframework.stereotype.Service;

import java.nio.file.Path;
import java.util.List;
import java.util.Map;
import java.util.Optional;
import java.util.concurrent.ConcurrentHashMap;

// Python equivalent: tasks.py — start_job(), run_pipeline(), active_jobs dict
@Service
public class JobService {

    @Autowired private JobRepository jobRepository;
    @Autowired private PipelineRunner pipelineRunner;
    @Autowired private UsageService usageService;

    @Value("${app.openrouter-api-key:}")
    private String backendApiKey;

    // In-memory progress cache — same purpose as active_jobs dict in Python tasks.py.
    // ConcurrentHashMap is thread-safe without a manual lock (replaces Python's threading.Lock).
    private final Map<String, Map<String, Object>> activeJobs = new ConcurrentHashMap<>();

    public List<Job> listJobs(String userId, int limit) {
        return jobRepository.findByUserIdOrderByCreatedAtDesc(userId, PageRequest.of(0, limit));
    }

    public Optional<Job> getJob(String jobId, String userId) {
        return jobRepository.findByIdAndUserId(jobId, userId);
    }

    public Job createJob(User user, Document document, String filename, Map<String, Object> settings) {
        Job job = Job.builder()
                .user(user)
                .document(document)
                .filename(filename)
                .settings(settings)
                .status(JobStatus.PENDING)
                .currentStep("Queued")
                .build();
        return jobRepository.save(job);
    }

    public void updateProgress(String jobId, double progress, String step, String detail) {
        jobRepository.findById(jobId).ifPresent(job -> {
            job.setProgress(progress);
            job.setCurrentStep(step);
            job.setStepDetail(detail);
            jobRepository.save(job);
        });
        activeJobs.computeIfPresent(jobId, (k, v) -> {
            v.put("progress", progress);
            v.put("currentStep", step);
            v.put("stepDetail", detail);
            return v;
        });
    }

    public void markCompleted(String jobId, String outputPdf, String outputPdfDriveId) {
        jobRepository.findById(jobId).ifPresent(job -> {
            job.setStatus(JobStatus.COMPLETED);
            job.setOutputPdf(outputPdf);
            job.setOutputPdfDriveId(outputPdfDriveId);
            jobRepository.save(job);
        });
        activeJobs.remove(jobId);
    }

    public void markFailed(String jobId, String error) {
        jobRepository.findById(jobId).ifPresent(job -> {
            job.setStatus(JobStatus.FAILED);
            job.setError(error);
            jobRepository.save(job);
        });
        activeJobs.remove(jobId);
    }

    public void cancel(String jobId) {
        jobRepository.findById(jobId).ifPresent(job -> {
            if (job.getStatus() == JobStatus.PENDING || job.getStatus() == JobStatus.PROCESSING) {
                job.setStatus(JobStatus.CANCELLED);
                job.setError("Cancelled by user");
                jobRepository.save(job);
            }
        });
    }

    // Returns live progress from the in-memory map, or null if the job is not running.
    // Controllers fall back to the DB-persisted values when this returns null.
    public Map<String, Object> getLiveStatus(String jobId) {
        return activeJobs.get(jobId);
    }

    // @Async dispatches this method to the ThreadPoolTaskExecutor defined in AppConfig.
    // Python equivalent: threading.Thread(target=run_pipeline, ...).start()
    @Async("taskExecutor")
    public void startPipeline(String jobId, Path pdfPath, Map<String, Object> settings) {
        activeJobs.put(jobId, new ConcurrentHashMap<>(Map.of(
                "progress", 0.0,
                "currentStep", "Starting",
                "stepDetail", "Initializing pipeline..."
        )));
        pipelineRunner.run(jobId, pdfPath, settings, this);
    }
}
