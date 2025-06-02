<script setup>
import { ref } from 'vue';
import SetupForm from './components/SetupForm.vue';
import PodcastDisplay from './components/PodcastDisplay.vue';
import AudioControls from './components/AudioControls.vue'; // 1. Import AudioControls

const currentPodcastData = ref(null);
const conversationLog = ref([]);
const generatedAudioFile = ref(null); // This will hold the filename like "podcast_xyz.mp3"

const handlePodcastGenerated = (apiResponseData) => {
  console.log("App.vue: Received podcast-generated event", apiResponseData);
  currentPodcastData.value = apiResponseData;
  conversationLog.value = apiResponseData.conversation_history || [];
  generatedAudioFile.value = apiResponseData.generated_audio_file || null;
  // If there's a new audio file, AudioControls will pick it up via the prop.
};

const handleNewDoubtResponse = (apiResponseData) => {
  console.log("App.vue: Received new-doubt-response event", apiResponseData);
  currentPodcastData.value = apiResponseData;
  conversationLog.value = apiResponseData.conversation_history || [];
  generatedAudioFile.value = apiResponseData.generated_audio_file || null;
  // This will also update the audio file for AudioControls if a new one is generated
}
</script>

<template>
  <div id="app-container">
    <h1>GenAI Podcast Studio</h1>

    <SetupForm @podcast-generated="handlePodcastGenerated" />

    <PodcastDisplay :conversationHistory="conversationLog" />

    <AudioControls 
      v-if="generatedAudioFile" 
      :audioFileName="generatedAudioFile" 
    /> {/* 2. Add AudioControls component and pass the prop */}

    </div>
</template>

<style scoped>
#app-container {
  max-width: 900px; /* Slightly wider to accommodate more content */
  margin: 2rem auto;
  padding: 1rem;
  font-family: sans-serif;
}
h1 {
  text-align: center;
  margin-bottom: 2rem;
  color: #e0e0e0; /* Lighter text for dark theme */
}
</style>