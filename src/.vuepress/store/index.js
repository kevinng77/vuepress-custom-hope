import {createStore} from 'vuex'


const store = createStore({
    state() {
        return {
            messages: [
                {"user":"bot","content":"Hi this is a test chatbot."},]
        }
    },
    mutations: {
        appendMessage: (state, message) => {
            state.messages.push(
                message
            )
        },
        updateResponse: (state, new_tokens) => {
            state.messages.slice(-1)[0].content += new_tokens;
        }
    },
    actions: {
        appendMessage: (context, message) => {
            context.commit('appendMessage', message)
        },
        updateResponse: (context, message) => {
            context.commit('updateResponse', message)
        }
    }

})

export default store;