<script setup>
import { ref, defineEmits, computed } from 'vue';
import axios from 'axios';

// Define the events that this component can emit
const emit = defineEmits(['podcast-generated', 'document-processed']); // Added document-processed for completeness

// --- Component State ---
const initialInformation = ref('');
const userThemes = ref('');
const generateAudio = ref(false); // We'll keep this for later use with bulk mode

// NEW: State for the two-stage form
const formStage = ref('initial'); // 'initial' or 'approval'
const proposedThemes = ref([]);
const currentThreadId = ref(null);
const currentDocId = ref(null); // Keep track of the docId within the form

const isLoading = ref(false);
const apiError = ref(null);


// --- File Upload Handling ---
const handleFileChange = async (event) => {
    const file = event.target.files[0];
    if (!file) return;

    isLoading.value = true;
    apiError.value = null;
    const docId = `doc_${Date.now()}`;
    const formData = new FormData();
    formData.append('file', file);
    formData.append('doc_id', docId);

    try {
        await axios.post('http://127.0.0.1:8000/process_document_for_rag', formData);
        currentDocId.value = docId;
        emit('document-processed', docId);
        apiError.value = null; // Clear previous errors
    } catch (error) {
        console.error("Error uploading document:", error);
        apiError.value = error.response?.data?.detail || "Failed to process document.";
    } finally {
        isLoading.value = false;
    }
};

// --- STAGE 1: Propose Themes ---
const handleProposeThemes = async () => {
    if (!currentDocId.value) {
        apiError.value = "Please upload a PDF document first.";
        return;
    }
    apiError.value = null;
    isLoading.value = true;

    const themesArray = userThemes.value
        .split(',')
        .map(theme => theme.trim())
        .filter(theme => theme.length > 0);

    const payload = {
        initial_information: initialInformation.value,
        user_provided_themes: themesArray.length > 0 ? themesArray : null,
        doc_id: currentDocId.value,
    };

    try {
        // Call the new endpoint
        const response = await axios.post('http://127.0.0.1:8000/propose_themes', payload);
        
        // Save the response and move to the next stage
        proposedThemes.value = response.data.themes;
        currentThreadId.value = response.data.thread_id;
        formStage.value = 'approval';

    } catch (error) {
        console.error("Error proposing themes:", error);
        apiError.value = error.response?.data?.detail || 'Failed to get themes.';
    } finally {
        isLoading.value = false;
    }
};

// --- STAGE 2: Initiate Podcast with Confirmed Themes ---
const handleInitiatePodcast = async () => {
    apiError.value = null;
    isLoading.value = true;

    // The themes from the textarea are joined by newlines. We split them back into an array.
    const finalThemesArray = proposedThemes.value.join('\n').split('\n').filter(t => t.trim().length > 0);

    const payload = {
        final_themes: finalThemesArray,
        doc_id: currentDocId.value,
        initial_information: initialInformation.value,
        generate_audio: generateAudio.value,
    };

    try {
        // Call the modified endpoint with the thread_id in the URL
        const response = await axios.post(`http://127.0.0.1:8000/initiate_podcast_flow/${currentThreadId.value}`, payload);
        
        // Emit the full response object which now contains the state and the original thread_id
        emit('podcast-generated', {
            thread_id: currentThreadId.value,
            state: response.data 
        });

    } catch (error) {
        console.error("Error initiating podcast:", error);
        apiError.value = error.response?.data?.detail || 'Failed to start podcast flow.';
    } finally {
        isLoading.value = false;
    }
};

// Helper to convert themes array to a string for the textarea
const themesAsText = computed({
    get: () => proposedThemes.value.join('\n'),
    set: (value) => {
        proposedThemes.value = value.split('\n');
    }
});

</script>

<template>
    <div class="setup-form">
        <!-- STAGE 1: Initial Setup -->
        <form @submit.prevent="handleProposeThemes" v-if="formStage === 'initial'">
            <h2>1. Setup Your Topic</h2>
            <div class="form-group">
                <label for="file-upload">Upload Document (PDF):</label>
                <input type="file" id="file-upload" @change="handleFileChange" accept=".pdf" required />
                <div v-if="currentDocId" class="doc-success">Document processed successfully!</div>
            </div>

            <div class="form-group">
                <label for="initial-info">Podcast Topic / Main Idea:</label>
                <textarea id="initial-info" v-model="initialInformation" rows="4" required></textarea>
            </div>

            <div class="form-group">
                <label for="user-themes">Optional: Provide Your Own Themes (comma-separated)</label>
                <input type="text" id="user-themes" v-model="userThemes" placeholder="e.g., history of the topic, future implications"/>
            </div>

            <button type="submit" :disabled="isLoading || !currentDocId">
                {{ isLoading ? 'Processing...' : 'Propose Guiding Themes' }}
            </button>
        </form>

        <!-- STAGE 2: Theme Approval -->
        <form @submit.prevent="handleInitiatePodcast" v-if="formStage === 'approval'">
            <h2>2. Approve Guiding Themes</h2>
            <p>You can edit the AI-proposed themes below before starting the podcast.</p>
            <div class="form-group">
                <label for="final-themes">Guiding Themes (one per line):</label>
                <textarea id="final-themes" v-model="themesAsText" rows="6"></textarea>
            </div>

            <div class="form-group-inline">
                <input type="checkbox" id="generate-audio" v-model="generateAudio" />
                <label for="generate-audio">Generate Audio at the End? (Future Feature)</label>
            </div>

            <button type="submit" :disabled="isLoading">
                {{ isLoading ? 'Starting...' : 'Confirm & Start Podcast' }}
            </button>
        </form>

        <div v-if="apiError" class="status-message error">
            Error: {{ apiError }}
        </div>
    </div>
</template>

<style scoped>
.setup-form {
    background-color: #2f2f2f;
    padding: 25px;
    border-radius: 8px;
    margin-bottom: 20px;
}
.form-group {
    margin-bottom: 20px;
    text-align: left;
}
.form-group-inline {
    margin-bottom: 20px;
    text-align: left;
    display: flex;
    align-items: center;
}
.form-group-inline input[type="checkbox"] {
    margin-right: 10px;
    width: auto;
}
label {
    display: block;
    margin-bottom: 8px;
    font-weight: bold;
    color: #ccc;
}
textarea, input[type="text"], input[type="file"] {
    width: 100%;
    padding: 10px;
    border-radius: 4px;
    border: 1px solid #555;
    background-color: #3b3b3b;
    color: #f0f0f0;
    font-size: 1em;
    box-sizing: border-box;
}
textarea {
    resize: vertical;
    font-family: inherit;
}
button {
    padding: 12px 25px;
    background-color: #007bff;
    color: white;
    border: none;
    border-radius: 4px;
    cursor: pointer;
    font-size: 1.1em;
    font-weight: bold;
    width: 100%;
}
button:hover:not(:disabled) {
    background-color: #0056b3;
}
button:disabled {
    background-color: #555;
    cursor: not-allowed;
}
.status-message {
    margin-top: 15px;
    padding: 10px;
    border-radius: 4px;
    text-align: center;
}
.error {
    background-color: #ffdddd;
    border: 1px solid #ff0000;
    color: #d8000c;
}
.doc-success {
    color: #28a745;
    font-weight: bold;
    margin-top: 8px;
}
h2 {
    border-bottom: 1px solid #444;
    padding-bottom: 10px;
    margin-bottom: 20px;
}
</style>