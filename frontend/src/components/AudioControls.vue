<script setup>
import { defineProps, computed, ref, watch, nextTick } from 'vue';

const props = defineProps({
  audioFileName: {
    type: String,
    default: null
  }
});

const BACKEND_BASE_URL = 'http://127.0.0.1:8000';
const audioElement = ref(null);

const audioSrc = computed(() => {
  if (props.audioFileName) {
    const url = `${BACKEND_BASE_URL}/get_podcast_audio/${props.audioFileName}`;
    console.log("AudioControls.vue: Constructed audio URL:", url);
    return url;
  }
  return null;
});

watch(audioSrc, async (newUrl) => {
  if (audioElement.value && newUrl) {
    await nextTick();
    audioElement.value.load();
    console.log("AudioControls.vue: Watcher triggered, audioElement.load() called.");
  }
});
</script>

<template>
  <div class="audio-controls" v-if="audioSrc">
    <h3>Podcast Audio</h3>
    <audio ref="audioElement" controls :src="audioSrc" controlslist="nodownload">
      Your browser does not support the audio element.
    </audio>
  </div>
</template>

<style scoped>
.audio-controls {
  margin-top: 2rem;
  padding: 20px;
  background-color: #2a2a2a;
  border-radius: 8px;
  text-align: center;
}
h3 {
  margin-top: 0;
  margin-bottom: 10px;
  color: #e0e0e0;
}
audio {
  width: 100%;
  margin-top: 5px;
}
</style>