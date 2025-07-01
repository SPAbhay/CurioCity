<script setup>
import { ref, defineProps, defineEmits } from 'vue';
import axios from 'axios';

const props = defineProps({
  threadId: {
    type: String,
    default: null
  },
  docId: {
    type: String,
    default: null
  },
  isPodcastActive: {
    type: Boolean,
    default: false
  }
});

const emit = defineEmits(['new-doubt-response', 'doubt-submission-error']);

const userDoubtText = ref('');
const isLoading = ref(false);

const handleSubmitDoubt = async () => {
  if (!props.threadId || !userDoubtText.value.trim()) {
    alert("Cannot submit an empty doubt or no active podcast session.");
    return;
  }

  isLoading.value = true;
  console.log(`Submitting doubt for thread ID: ${props.threadId}`);

  const payload = {
    user_doubt_text: userDoubtText.value,
    generate_audio: true // Always generate audio for doubt responses for now
  };

  try {
    const response = await axios.post(
      `http://127.0.0.1:8000/submit_doubt/${props.threadId}`, 
      payload
    );
    console.log("Doubt response received:", response.data);
    emit('new-doubt-response', response.data);
    userDoubtText.value = ''; // Clear the input field
  } catch (error) {
    console.error("Error submitting doubt:", error);
    const errorMessage = error.response ? error.response.data.detail : "Network error";
    emit('doubt-submission-error', errorMessage);
  } finally {
    isLoading.value = false;
  }
};
</script>

<template>
  <div class="interaction-controls" v-if="isPodcastActive">
    <h3>Ask a Doubt</h3>
    <p>Heard something you want to clarify? Type your question below to interrupt the podcast.</p>
    <form @submit.prevent="handleSubmitDoubt" class="doubt-form">
      <input
        type="text"
        v-model="userDoubtText"
        placeholder="e.g., What did you mean by 'spacetime curvature'?"
        :disabled="isLoading"
      />
      <button type="submit" :disabled="isLoading">
        {{ isLoading ? 'Submitting...' : 'Submit Doubt' }}
      </button>
    </form>
  </div>
</template>

<style scoped>
.interaction-controls {
  margin-top: 2rem;
  padding: 25px;
  background-color: #3a3a3a;
  border-radius: 8px;
  border: 1px solid #555;
}
h3, p {
  text-align: center;
}
p {
  font-size: 0.9em;
  color: #ccc;
  margin-bottom: 15px;
}
.doubt-form {
  display: flex;
  gap: 10px;
}
.doubt-form input[type="text"] {
  flex-grow: 1;
  padding: 8px 10px;
  border-radius: 4px;
  border: 1px solid #555;
  background-color: #2f2f2f;
  color: #f0f0f0;
  font-size: 1em;
}
.doubt-form button {
  padding: 8px 15px;
  background-color: #ff9900;
  color: black;
  border: none;
  border-radius: 4px;
  cursor: pointer;
  font-weight: bold;
  transition: background-color 0.2s;
}
.doubt-form button:hover {
  background-color: #ffaf3a;
}
.doubt-form button:disabled {
  background-color: #555;
  cursor: not-allowed;
}
</style>
