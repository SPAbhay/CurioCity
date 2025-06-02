<script setup>
import { defineProps, computed, ref, watch, onMounted, nextTick } from 'vue';

const props = defineProps({
  audioFileName: {
    type: String,
    default: null
  }
});

const BACKEND_BASE_URL = 'http://127.0.0.1:8000'; // JavaScript string constant
const audioElement = ref(null);

const audioSrc = computed(() => {
  console.log("AudioControls.vue: Computing audioSrc. props.audioFileName:", props.audioFileName);
  if (props.audioFileName && typeof props.audioFileName === 'string') { // Add type check for safety
    // Ensure template literal is correctly using backticks and interpolation
    const constructedUrl = `${BACKEND_BASE_URL}/get_podcast_audio/${props.audioFileName}`;
    console.log("AudioControls.vue: Constructed audio URL:", constructedUrl); // This should be the actual URL
    return constructedUrl;
  }
  console.log("AudioControls.vue: audioFileName is null or not a string, so audioSrc is null.");
  return null;
});

watch(audioSrc, async (newUrl, oldUrl) => {
  if (newUrl && audioElement.value) {
    console.log(`AudioControls.vue: audioSrc watcher - new URL: ${newUrl}. Calling load().`);
    await nextTick();
    audioElement.value.load();
  } else if (!newUrl) {
    console.log("AudioControls.vue: audioSrc watcher - URL is null.");
  }
});

onMounted(async () => {
  // This might not be strictly necessary if the watch effect handles the initial load well
  // due to audioSrc changing from its initial null (if props.audioFileName is present on mount).
  // However, it doesn't hurt to check.
  if (audioSrc.value && audioElement.value) {
    console.log("AudioControls.vue: Component mounted with audioSrc. Calling load().", audioSrc.value);
    await nextTick();
    audioElement.value.load();
  }
});

</script>

<template>
  <div class="audio-controls" v-if="audioSrc">
    <h3>Podcast Audio</h3>
    <audio ref="audioElement" controls :src="audioSrc" controlslist="nodownload">
      Your browser does not support the audio element.
    </audio>
    <p v-if="audioSrc">Playing: {{ audioSrc }}</p> </div>
  <div v-else>
    <p v-if="props.audioFileName">Audio player preparing for: {{ props.audioFileName }}...</p>
    </div>
</template>

<style scoped>
.audio-controls {
  margin-top: 20px;
  padding: 15px;
  background-color: #2a2a2a; /* Slightly different background */
  border-radius: 8px;
  text-align: center;
}
.audio-controls h3 {
  margin-top: 0;
  margin-bottom: 10px;
  color: #e0e0e0;
}
audio {
  width: 100%; /* Make the audio player responsive */
  margin-top: 5px;
}
/* Optional: Style the controls for dark theme if browser defaults aren't great */
/* audio::-webkit-media-controls-panel {
  background-color: #3b3b3b;
  color: #e0e0e0;
}
audio::-webkit-media-controls-play-button,
audio::-webkit-media-controls-volume-slider,
audio::-webkit-media-controls-mute-button,
audio::-webkit-media-controls-timeline,
audio::-webkit-media-controls-current-time-display,
audio::-webkit-media-controls-time-remaining-display {
  filter: invert(1) grayscale(1) brightness(1.5); 
} */
</style>