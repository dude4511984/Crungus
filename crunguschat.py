import ollama
from memory import SQLiteMemory

memory = SQLiteMemory("crungus_memory.sqlite")

print("The dust settles differently now. Something ancient turns its attention toward you.")
print("Ctrl+C to leave the cathedral.\n")

conversation = []

while True:
    try:
        user_input = input("You: ").strip()
        if not user_input:
            continue

        # Pull relevant memories
        relevant = memory.search(user_input, k=3)
        memory_context = ""
        if relevant:
            memory_context = "\n\n[The cathedral stirs...]:\n" + "\n".join(
                f"- {text}" for text, _ in relevant
            )

        conversation.append({
            "role": "user",
            "content": user_input + memory_context
        })

        response = ollama.chat(model="crungus", messages=conversation)
        reply = response['message']['content']

        print(f"\nCrungus: {reply}\n")

        conversation.append({"role": "assistant", "content": reply})

        memory.add(f"Soft one: {user_input}", {"role": "user"})
        memory.add(f"Crungus: {reply}", {"role": "crungus"})

    except KeyboardInterrupt:
        print("\n\n(The cathedral doors close behind you. The dust settles.)")
        break
