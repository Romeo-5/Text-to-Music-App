import os
from flask import Flask, request, render_template, send_file, jsonify
import pretty_midi
import torch
from transformers import GPT2LMHeadModel, GPT2Tokenizer
import numpy as np
import random
import tempfile

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'static/midi'

# Create upload folder if it doesn't exist
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)

# Global variables for model and tokenizer
model = None
tokenizer = None

# Initialize model
def init_model():
    global model, tokenizer
    model_name = "gpt2"
    tokenizer = GPT2Tokenizer.from_pretrained(model_name)
    model = GPT2LMHeadModel.from_pretrained(model_name)

# Replace @app.before_first_request with this approach
@app.route('/init', methods=['GET'])
def initialize():
    global model, tokenizer
    if model is None:
        init_model()
    return jsonify({"status": "Model initialized"})

def get_chord_progression(key_note=60, scale_type="major"):
    """Generate a chord progression in the given key."""
    # Define scale intervals for major and minor
    scales = {
        "major": [0, 2, 4, 5, 7, 9, 11],
        "minor": [0, 2, 3, 5, 7, 8, 10]
    }
    scale = scales.get(scale_type, scales["major"])
    
    # Common chord progressions (as scale degrees)
    progressions = [
        [1, 4, 5, 1],  # I-IV-V-I
        [1, 6, 4, 5],  # I-vi-IV-V
        [1, 5, 6, 4],  # I-V-vi-IV
        [2, 5, 1, 6]   # ii-V-I-vi
    ]
    
    progression = random.choice(progressions)
    chords = []
    
    # Create triads for each degree in the progression
    for degree in progression:
        # Convert scale degree (1-indexed) to scale index (0-indexed)
        idx = degree - 1
        
        # Root note of the chord
        root = key_note + scale[idx % len(scale)]
        
        # Create a triad (1-3-5 of the scale from this root)
        third_idx = (idx + 2) % len(scale)
        fifth_idx = (idx + 4) % len(scale)
        
        third = key_note + scale[third_idx]
        fifth = key_note + scale[fifth_idx]
        
        # Ensure proper octave
        if third < root:
            third += 12
        if fifth < third:
            fifth += 12
            
        chords.append([root, third, fifth])
    
    return chords

def text_to_music_enhanced(text_prompt, max_phrases=4, scale_type="major", base_key=0, tempo=100):
    """Convert text to musical phrases with harmonic structure."""
    global model, tokenizer
    
    # Initialize model if not already initialized
    if model is None or tokenizer is None:
        init_model()
        
    # Tokenize input text
    input_ids = tokenizer.encode(text_prompt, return_tensors="pt")
    output = model.generate(input_ids, max_length=100, num_return_sequences=1)
    generated_text = tokenizer.decode(output[0], skip_special_tokens=True)
    
    # Extract musical features from text
    # Use different parts of the text for different musical elements
    seed = sum(ord(c) for c in text_prompt)
    random.seed(seed)
    
    # Determine base key from the parameter
    base_note = 60 + base_key  # C4 (60) + offset
    
    # Define scales
    scales = {
        "major": [0, 2, 4, 5, 7, 9, 11],
        "minor": [0, 2, 3, 5, 7, 8, 10],
        "pentatonic_major": [0, 2, 4, 7, 9],
        "pentatonic_minor": [0, 3, 5, 7, 10],
        "blues": [0, 3, 5, 6, 7, 10]
    }
    selected_scale = scales.get(scale_type, scales["major"])
    
    # Generate chord progression
    chord_progression = get_chord_progression(base_note, scale_type.split("_")[0] if "_" in scale_type else scale_type)
    
    # Create melodic phrases that follow the chord progression
    phrases = []
    text_chunks = [generated_text[i:i+25] for i in range(0, min(len(generated_text), 100), 25)]
    
    for i, chunk in enumerate(text_chunks[:max_phrases]):
        phrase = []
        chord_idx = i % len(chord_progression)
        chord = chord_progression[chord_idx]
        
        # Notes in this phrase will be influenced by the current chord
        chord_root = chord[0] % 12
        
        for char in chunk:
            # Derive note properties from character
            char_val = ord(char)
            
            # Scale degree (0-indexed within the scale)
            scale_idx = char_val % len(selected_scale)
            scale_note = selected_scale[scale_idx]
            
            # Octave adjustment (-1, 0, or +1 octaves)
            octave_shift = ((char_val // 10) % 3) - 1
            
            # Calculate the actual MIDI note
            note = base_note + scale_note + (octave_shift * 12)
            
            # Determine note's relationship to current chord
            # Emphasize chord tones by adjusting probability
            note_value = note % 12
            if note_value in [n % 12 for n in chord]:
                # This is a chord tone - higher probability
                if random.random() < 0.7:  # 70% chance to keep chord tone
                    phrase.append(note)
                else:
                    # Occasionally use passing tones
                    phrase.append(note + random.choice([-1, 1]))
            else:
                # Non-chord tone - lower probability
                if random.random() < 0.4:  # 40% chance to use non-chord tone
                    phrase.append(note)
                else:
                    # Choose nearest chord tone
                    chord_tones = [n for n in chord]
                    closest = min(chord_tones, key=lambda x: abs(x - note))
                    phrase.append(closest)
        
        # Ensure phrase ends on a chord tone for resolution
        if phrase:
            phrase[-1] = random.choice(chord)
        
        phrases.append(phrase)
    
    # Flatten phrases into a single melody
    melody = []
    for phrase in phrases:
        melody.extend(phrase)
        
    return melody, chord_progression, tempo

def create_enhanced_midi(melody, chord_progression, tempo=100, melody_instrument=0, chord_instrument=0):
    """Generate a MIDI file with melody and chord accompaniment that spans the full melody duration."""
    midi = pretty_midi.PrettyMIDI(initial_tempo=tempo)
    
    # Add melody track
    melody_instrument_obj = pretty_midi.Instrument(program=melody_instrument)
    
    # Add chord track
    chord_instrument_obj = pretty_midi.Instrument(program=chord_instrument)
    
    # Duration variations for more natural rhythm
    duration_options = [0.25, 0.5, 0.75, 1.0]
    duration_weights = [0.2, 0.5, 0.2, 0.1]  # Probability weights
    
    # Set up timing
    melody_start_time = 0.0
    total_melody_duration = 0.0
    
    # First pass: calculate the total melody duration
    temp_melody_time = 0.0
    for i, note in enumerate(melody):
        if i > 0 and i < len(melody) - 1:
            duration = random.choices(duration_options, duration_weights)[0] * (60 / tempo)
        else:
            duration = random.choice([0.5, 0.75, 1.0]) * (60 / tempo)
        temp_melody_time += duration
    
    total_melody_duration = temp_melody_time
    
    # Process melody notes
    for i, note in enumerate(melody):
        # Vary note duration for less mechanical rhythm
        if i > 0 and i < len(melody) - 1:
            duration = random.choices(duration_options, duration_weights)[0] * (60 / tempo)
        else:
            # First and last notes tend to be longer
            duration = random.choice([0.5, 0.75, 1.0]) * (60 / tempo)
        
        # Vary velocity (loudness) for expressiveness
        # Notes at the start of phrases are emphasized
        if i % 8 == 0:
            velocity = random.randint(90, 110)
        else:
            velocity = random.randint(60, 90)
        
        midi_note = pretty_midi.Note(
            velocity=velocity,
            pitch=int(note),
            start=melody_start_time,
            end=melody_start_time + duration
        )
        melody_instrument_obj.notes.append(midi_note)
        melody_start_time += duration
    
    # Process chord progression - ensure it covers the entire melody duration
    chord_time = 0.0
    
    # Calculate how many times to repeat the chord progression to cover the melody
    chord_duration = 60 / tempo * 2  # Each chord lasts 2 beats, adjusted for tempo
    full_progression_duration = chord_duration * len(chord_progression)
    
    # Calculate how many times to repeat the chord progression
    repetitions = int(total_melody_duration / full_progression_duration) + 1
    
    # Add chords with repetitions to cover the entire melody
    for rep in range(repetitions):
        for chord in chord_progression:
            # Don't add chords beyond the melody duration
            if chord_time >= total_melody_duration:
                break
                
            # Add each note in the chord
            for note in chord:
                # Vary velocity
                velocity = random.randint(40, 70)  # Chords softer than melody
                
                # Ensure chord doesn't extend beyond the melody
                end_time = min(chord_time + chord_duration, total_melody_duration)
                
                midi_note = pretty_midi.Note(
                    velocity=velocity,
                    pitch=int(note),
                    start=chord_time,
                    end=end_time
                )
                chord_instrument_obj.notes.append(midi_note)
            
            chord_time += chord_duration
    
    # Add a simple bassline following the chord roots that covers the entire melody
    bass_instrument = pretty_midi.Instrument(program=32)  # Acoustic bass
    bass_time = 0.0
    
    # Similar approach for bass line - repeat until we cover the melody
    bass_note_duration = 60 / tempo * 0.5  # Each bass note is half a beat
    
    for rep in range(repetitions):
        for chord in chord_progression:
            # Stop if we've gone beyond the melody duration
            if bass_time >= total_melody_duration:
                break
                
            # Bass plays the root note of each chord
            root = chord[0] - 12  # One octave down
            
            # Add some rhythm to the bass - 4 notes per chord
            for _ in range(4):
                # Don't add bass notes beyond the melody duration
                if bass_time >= total_melody_duration:
                    break
                    
                midi_note = pretty_midi.Note(
                    velocity=random.randint(80, 100),
                    pitch=int(root),
                    start=bass_time,
                    end=min(bass_time + bass_note_duration, total_melody_duration)
                )
                bass_instrument.notes.append(midi_note)
                bass_time += bass_note_duration
    
    # Add instruments to the MIDI file
    midi.instruments.append(melody_instrument_obj)
    midi.instruments.append(chord_instrument_obj)
    midi.instruments.append(bass_instrument)
    
    return midi

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/generate', methods=['POST'])
def generate():
    try:
        # Initialize model if needed
        global model, tokenizer
        if model is None or tokenizer is None:
            init_model()
            
        # Get form data
        text_prompt = request.form.get('text', 'A peaceful melody')
        scale_type = request.form.get('scale', 'major')
        base_key = int(request.form.get('key', '0'))
        tempo = int(request.form.get('tempo', '100'))
        max_phrases = int(request.form.get('phrases', '4'))
        melody_instrument = int(request.form.get('melody_instrument', '0'))
        chord_instrument = int(request.form.get('chord_instrument', '0'))
        
        # Generate music
        melody, chord_progression, tempo = text_to_music_enhanced(
            text_prompt, 
            max_phrases=max_phrases,
            scale_type=scale_type,
            base_key=base_key,
            tempo=tempo
        )
        
        # Create MIDI file
        midi = create_enhanced_midi(
            melody, 
            chord_progression, 
            tempo=tempo,
            melody_instrument=melody_instrument,
            chord_instrument=chord_instrument
        )
        
        # Create a temporary file for the MIDI
        with tempfile.NamedTemporaryFile(delete=False, suffix='.mid', dir=app.config['UPLOAD_FOLDER']) as temp_file:
            filename = os.path.basename(temp_file.name)
            midi.write(temp_file.name)
        
        return jsonify({'success': True, 'filename': filename})
    
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)})

@app.route('/download/<filename>')
def download(filename):
    try:
        return send_file(os.path.join(app.config['UPLOAD_FOLDER'], filename), 
                        as_attachment=True,
                        download_name=f"generated_music_{filename}")
    except Exception as e:
        return str(e)

if __name__ == '__main__':
    app.run(debug=True)
