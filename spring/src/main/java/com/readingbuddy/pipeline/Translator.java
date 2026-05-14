package com.readingbuddy.pipeline;

import com.fasterxml.jackson.databind.JsonNode;
import com.fasterxml.jackson.databind.ObjectMapper;
import org.springframework.stereotype.Component;

import java.net.URI;
import java.net.http.HttpClient;
import java.net.http.HttpRequest;
import java.net.http.HttpResponse;
import java.time.Duration;
import java.util.List;
import java.util.Map;
import java.util.function.BiConsumer;

@Component
public class Translator {

    private static final String OPENROUTER_URL = "https://openrouter.ai/api/v1/chat/completions";

    private static final String SYSTEM_PROMPT =
            "You are a professional translator. Translate the following text to Sinhala (සිංහල). "
            + "Preserve the original structure, headings, and paragraphs. "
            + "Return only the translated text without any additional commentary.";

    private final HttpClient httpClient = HttpClient.newBuilder()
            .connectTimeout(Duration.ofSeconds(30))
            .build();

    private final ObjectMapper mapper = new ObjectMapper();

    public List<PageData> translate(List<PageData> pages, String apiKey, String model,
                                    double temperature, BiConsumer<Double, String> progress)
            throws Exception {

        int total = pages.size();

        for (int i = 0; i < total; i++) {
            PageData page = pages.get(i);
            String text = page.getDigitalText();

            String translated = text.isBlank() ? "" : callOpenRouter(text, apiKey, model, temperature);

            page.setTranslatedText(translated);
            page.setTranslatedChars(translated.length());

            double pct = ((double) (i + 1) / total) * 100;
            progress.accept(pct,
                    String.format("Translating page %d/%d", page.getPageNumber(), total));
        }

        return pages;
    }

    private String callOpenRouter(String text, String apiKey, String model,
                                  double temperature) throws Exception {

        String body = mapper.writeValueAsString(Map.of(
                "model", model,
                "temperature", temperature,
                "messages", List.of(
                        Map.of("role", "system", "content", SYSTEM_PROMPT),
                        Map.of("role", "user",   "content", text)
                )
        ));

        HttpRequest request = HttpRequest.newBuilder()
                .uri(URI.create(OPENROUTER_URL))
                .header("Authorization", "Bearer " + apiKey)
                .header("Content-Type", "application/json")
                .timeout(Duration.ofSeconds(120))
                .POST(HttpRequest.BodyPublishers.ofString(body))
                .build();

        HttpResponse<String> response = httpClient.send(request, HttpResponse.BodyHandlers.ofString());

        if (response.statusCode() != 200) {
            throw new RuntimeException("OpenRouter API error: HTTP " + response.statusCode()
                    + " — " + response.body());
        }

        JsonNode root = mapper.readTree(response.body());
        return root.path("choices").path(0).path("message").path("content").asText();
    }
}
