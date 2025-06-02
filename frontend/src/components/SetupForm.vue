<script setup>
import { ref, defineEmits } from 'vue'; 
import axios from 'axios';

// Define the event that this component can emit
const emit = defineEmits(['podcast-generated']);

const initialInformation = ref('');
const userThemes = ref(''); 
const generateAudio = ref(false);

const isLoading = ref(false);
const apiError = ref(null);
// const podcastData = ref(null); // We can remove this if App.vue now manages the state

const handleSubmit = async () => {
  console.log("Form Submitted!");
  apiError.value = null;
  isLoading.value = true;
  // podcastData.value = null; // Clear previous data

  const themesArray = userThemes.value
    .split(',')
    .map(theme => theme.trim())
    .filter(theme => theme.length > 0);

  const payload = {
    initial_information: initialInformation.value,
    user_provided_themes: themesArray.length > 0 ? themesArray : null,
    generate_audio: generateAudio.value
  };

  console.log("Sending payload to backend:", payload);

  try {
    const response = await axios.post('http://127.0.0.1:8000/initiate_podcast_flow', payload);

    console.log("Backend Response (AgentState):", response.data);
    emit('podcast-generated', response.data); // Emit event with the response data

  } catch (error) {
    console.error("Error calling API:", error);
    if (error.response) {
      apiError.value = error.response.data.detail || 'Error from server.';
    } else if (error.request) {
      apiError.value = 'No response from server. Is the backend running?';
    } else {
      apiError.value = error.message;
    }
  } finally {
    isLoading.value = false;
  }
};
</script>

<template>
  <div class="setup-form">
    <h2>Create Your Podcast</h2>
    <form @submit.prevent="handleSubmit">
      <div class="form-group">
        <label for="initial-info">Initial Topic/Information:</label>
        <textarea id="initial-info" v-model="initialInformation" rows="5" required></textarea>
      </div>

      <div class="form-group">
        <label for="user-themes">Optional Guiding Themes (comma-separated):</label>
        <input type="text" id="user-themes" v-model="userThemes" />
      </div>

      <div class="form-group-inline">
        <input type="checkbox" id="generate-audio" v-model="generateAudio" />
        <label for="generate-audio">Generate Audio Output?</label>
      </div>

      <button type="submit" :disabled="isLoading">
        {{ isLoading ? 'Generating...' : 'Start Podcast' }}
      </button>
    </form>

    <div v-if="isLoading" class="status-message">Generating podcast, please wait...</div>
    <div v-if="apiError" class="status-message error">
      Error: {{ apiError.detail || apiError }}
    </div>

    </div>
</template>

<style scoped>
/* ... your existing styles ... */
.setup-form {
  background-color: #2f2f2f;
  padding: 20px;
  border-radius: 8px;
  margin-bottom: 20px;
}
.form-group {
  margin-bottom: 15px;
  text-align: left;
}
.form-group-inline {
  margin-bottom: 15px;
  text-align: left;
  display: flex;
  align-items: center;
}
.form-group-inline input[type="checkbox"] {
  margin-right: 8px;
  width: auto; 
}
label {
  display: block;
  margin-bottom: 5px;
  font-weight: bold;
}
textarea, input[type="text"] {
  width: calc(100% - 22px); /* Adjusted for border and padding */
  padding: 8px 10px;
  border-radius: 4px;
  border: 1px solid #555;
  background-color: #3b3b3b;
  color: #f0f0f0;
  font-size: 1em;
  box-sizing: border-box; /* Ensures padding and border are inside the width */
}
textarea {
  resize: vertical;
}
button {
  padding: 10px 20px;
  background-color: #007bff;
  color: white;
  border: none;
  border-radius: 4px;
  cursor: pointer;
  font-size: 1em;
}
button:hover {
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
}
.error {
  background-color: #ffdddd;
  border: 1px solid #ff0000;
  color: #d8000c;
}
</style>