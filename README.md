#LlamaPD

Although police departments have spent millions buying body cameras, manual review is so labor-intensive and subjective that footage is rarely processed. Experts, police, and citizen activists alike all recognize that body cameras improve transparency, accountability, and safety. Enter LlamaPD, a Llama-powered system that automatically analyzes transcripts of police body cam footage to deliver efficient, private, and objective analysis.

##Features
<ul>
  <li>Automatically analyze audio from video footage from police body cams, no matter how long.</li>
  <li>Run it locally, privately, and efficiently, even without WiFi.</li>
  <li>Add police procedure manuals that the AI can automatically reference and learn from.</li>
</ul>


To start running, open a Docker session and then start with:
<code>docker compose build</code>

<code>docker compose up -d</code>

Then, simply open http://localhost:5001/ to access the web portal, where you can upload video files for analysis, which will display on the website.


For debugging purposes, you can enter the backend environment via Bash using the following command:
<code>docker exec -it backend /bin/bash</code>
