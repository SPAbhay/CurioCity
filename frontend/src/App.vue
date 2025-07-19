<script setup>
import { ref, computed } from 'vue';
import axios from 'axios';
import SetupForm from './components/SetupForm.vue';
import PodcastDisplay from './components/PodcastDisplay.vue';
import AudioControls from './components/AudioControls.vue';
import InteractionControls from './components/InteractionControls.vue';
import ThemeTracker from './components/ThemeTracker.vue'; // 1. Import the new component

// --- State Management ---
const currentPodcastState = ref(null);
const currentThreadId = ref(null);
const currentDocId = ref(null);
const isLoading = ref(false);
const generationMode = ref(null);

// --- Computed Properties for Child Components ---
const conversationLog = computed(() => currentPodcastState.value?.conversation_history || []);
const generatedAudioFile = computed(() => currentPodcastState.value?.generated_audio_file || null);
const isPodcastActive = computed(() => currentThreadId.value !== null);

// NEW: Computed properties to pass to the ThemeTracker
const guidingThemes = computed(() => currentPodcastState.value?.guiding_themes || []);
const coveredThemes = computed(() => currentPodcastState.value?.covered_themes || []);


// --- Event Handlers ---
const handleDocumentProcessed = (docId) => {
  console.log("App.vue: Received 'document-processed' event with doc_id:", docId);
  currentDocId.value = docId;
  currentPodcastState.value = null;
  currentThreadId.value = null;
};

const handlePodcastGenerated = (apiResponse) => {
  console.log("App.vue: Received 'podcast-generated' event with payload:", apiResponse);
  currentThreadId.value = apiResponse.thread_id;
  currentPodcastState.value = apiResponse.state;
  generationMode.value = apiResponse.state.generation_mode;
};

const handleNewDoubtResponse = (updatedState) => {
  console.log("App.vue: Received 'new-doubt-response' event with updated state:", updatedState);
  currentPodcastState.value = updatedState;
};

const handleDoubtError = (errorMessage) => {
  console.error("App.vue: Received 'doubt-submission-error' event:", errorMessage);
  alert(`Error submitting your doubt: ${errorMessage}`);
}

const handleContinue = async () => {
    if (!currentThreadId.value) {
        console.error("Cannot continue, threadId is missing.");
        return;
    }
    isLoading.value = true;
    try {
        const response = await axios.post(`http://127.0.0.1:8000/continue_flow/${currentThreadId.value}`);
        const newState = response.data;
        console.log("App.vue: Received new state after continue", newState);
        currentPodcastState.value = newState;
    } catch (error) {
        console.error("Error calling continue API:", error);
        alert(`Failed to get next turn: ${error.response?.data?.detail || error.message}`);
    } finally {
        isLoading.value = false;
    }
};
</script>

<template>
  <div id="app-container">
    <header class="app-header">
      <h1>GenAI Podcast Studio</h1>
      <p class="subtitle">Upload a document, define a topic, and listen to AI hosts discuss its contents.</p>
    </header>
    
    <SetupForm 
      @document-processed="handleDocumentProcessed"
      @podcast-generated="handlePodcastGenerated" 
    />

    <!-- 2. Add the new component here -->
    <ThemeTracker
      v-if="isPodcastActive"
      :guiding-themes="guidingThemes"
      :covered-themes="coveredThemes"
    />
    
    <PodcastDisplay :conversationHistory="conversationLog" />

    <div class="controls" v-if="currentPodcastState && currentPodcastState.current_turn > 0 && generationMode === 'interactive'">
        <button @click="handleContinue" :disabled="isLoading">
            {{ isLoading ? 'Generating...' : 'Next Turn' }}
        </button>
    </div>

    <AudioControls 
        v-if="generatedAudioFile" 
        :audioFileName="generatedAudioFile" 
    />

    <InteractionControls 
      v-if="isPodcastActive && conversationLog.length > 0 && generationMode === 'interactive'"
      :isPodcastActive="isPodcastActive"
      :threadId="currentThreadId"
      :docId="currentDocId"
      @new-doubt-response="handleNewDoubtResponse"
      @doubt-submission-error="handleDoubtError"
    />
  </div>
</template>

<style scoped>
#app-container {
  max-width: 900px;
  margin: 0 auto;
  padding: 2rem;
  font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, Helvetica, Arial, sans-serif;
}
.app-header {
  text-align: center;
  margin-bottom: 2.5rem;
  border-bottom: 1px solid #444;
  padding-bottom: 1.5rem;
}
h1 {
  font-size: 2.5em;
  font-weight: 700;
  color: #e0e0e0;
  margin: 0;
}
.subtitle {
  font-size: 1.1em;
  color: #aaa;
  margin-top: 0.5rem;
}
.controls {
  text-align: center;
  margin: 20px 0;
}
.controls button {
    padding: 10px 25px;
    font-size: 1.1em;
    cursor: pointer;
    background-color: #007bff;
    color: white;
    border: none;
    border-radius: 5px;
    transition: background-color 0.2s;
}
.controls button:hover:not(:disabled) {
    background-color: #0056b3;
}
.controls button:disabled {
    background-color: #555;
    cursor: not-allowed;
}
</style>