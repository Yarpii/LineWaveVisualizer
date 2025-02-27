import pygame
from pygame.locals import *
from OpenGL.GL import *
from OpenGL.GL.shaders import compileProgram, compileShader
import numpy as np
import time
import scipy.signal as signal

# Vertex shader remains the same as before
VERTEX_SHADER = """
#version 330
in vec2 position;
uniform float time;
uniform float frequency;
uniform float amplitude;
uniform float lineOffset;  // Offset for line effect
uniform mat4 model;
uniform mat4 view;
uniform mat4 projection;

void main() {
    // Base wave formula
    float z = amplitude * sin(frequency * (position.x * position.x + position.y * position.y) - time)
            * cos(frequency * (position.x + position.y) - time/2.0);
    
    // Create offset effect for parallel lines
    float zOffset = z + lineOffset;
    
    // Final position
    gl_Position = projection * view * model * vec4(position.x, zOffset, position.y, 1.0);
}
"""

# Fragment shader remains the same as before
FRAGMENT_SHADER = """
#version 330
out vec4 fragColor;
uniform vec3 lineColor;
uniform float time;
uniform float frequency;

void main() {
    // Create subtle pulsing glow effect synchronized with frequency
    float pulse = 0.8 + 0.2 * sin(time * (frequency/440.0));
    vec3 color = lineColor * pulse;
    
    // Add subtle color variation
    color.r += 0.1 * sin(time * 0.3);
    color.g += 0.1 * sin(time * 0.5);
    color.b += 0.1 * sin(time * 0.7);
    
    fragColor = vec4(color, 1.0);
}
"""

def create_rotation_matrix(angle, axis):
    """Creates a rotation matrix for a given angle and axis."""
    axis = axis / np.linalg.norm(axis)
    a = np.cos(angle / 2.0)
    b, c, d = -axis * np.sin(angle / 2.0)
    
    return np.array([
        [a*a+b*b-c*c-d*d, 2*(b*c-a*d), 2*(b*d+a*c), 0],
        [2*(b*c+a*d), a*a+c*c-b*b-d*d, 2*(c*d-a*b), 0],
        [2*(b*d-a*c), 2*(c*d+a*b), a*a+d*d-b*b-c*c, 0],
        [0, 0, 0, 1]
    ], dtype=np.float32)

def perspective(fovy, aspect, near, far):
    """Creates a perspective projection matrix."""
    f = 1.0 / np.tan(np.radians(fovy) / 2)
    proj = np.zeros((4, 4), dtype=np.float32)
    proj[0, 0] = f / aspect
    proj[1, 1] = f
    proj[2, 2] = (far + near) / (near - far)
    proj[2, 3] = (2 * far * near) / (near - far)
    proj[3, 2] = -1.0
    return proj

def lookAt(eye, center, up):
    """Creates a view matrix using the lookAt convention with special handling for top-down view."""
    f = center - eye
    f_length = np.linalg.norm(f)
    
    if f_length < 1e-6:  # Handle case when eye and center are very close
        f = np.array([0, 0, -1], dtype=np.float32)
    else:
        f = f / f_length
    
    s = np.cross(f, up)
    s_length = np.linalg.norm(s)
    
    if s_length < 1e-6:  # Handle case when f and up are parallel (top-down view)
        # Choose a different right vector
        if abs(up[1]) > 0.9:  # If up is mostly in Y direction
            s = np.array([1, 0, 0], dtype=np.float32)
        else:
            s = np.array([0, 1, 0], dtype=np.float32)
    else:
        s = s / s_length
    
    u = np.cross(s, f)
    
    M = np.eye(4, dtype=np.float32)
    M[0, :3] = s
    M[1, :3] = u
    M[2, :3] = -f
    
    T = np.eye(4, dtype=np.float32)
    T[0, 3] = -eye[0]
    T[1, 3] = -eye[1]
    T[2, 3] = -eye[2]
    
    return M @ T

class SoundGenerator:
    """A more sophisticated sound generator with multiple waveforms and effects."""
    
    def __init__(self, sample_rate=44100):
        self.sample_rate = sample_rate
        self.channels = {}  # Store multiple sound channels
        
        # Available waveform types
        self.waveforms = {
            'sine': self._sine_wave,
            'square': self._square_wave,
            'sawtooth': self._sawtooth_wave,
            'triangle': self._triangle_wave,
            'noise': self._noise
        }
        
        # Initialize pygame mixer
        pygame.mixer.init(frequency=self.sample_rate, size=-16, channels=2)
        
        # Default sound parameters
        self.current_waveform = 'sine'
        self.base_frequency = 440.0  # A4
        self.volume = 0.3
        self.stereo_pan = 0.5  # Center (0 = left, 1 = right)
        self.harmonics = [(1.0, 1.0), (2.0, 0.5), (3.0, 0.25)]  # (frequency ratio, amplitude)
        self.filter_freq = 1000  # Filter cutoff in Hz
        self.resonance = 1.0
        
        # Musical scale parameters
        self.scale_type = 'major'
        self.root_note = 'A'
        self.octave = 4
        
        # For chord generation
        self.chord_type = 'major'
        self.playing_chord = False
        
        # Notes frequency map (A4 = 440Hz)
        self.notes = {
            'C': 261.63, 'C#': 277.18, 'D': 293.66, 'D#': 311.13,
            'E': 329.63, 'F': 349.23, 'F#': 369.99, 'G': 392.00,
            'G#': 415.30, 'A': 440.00, 'A#': 466.16, 'B': 493.88
        }
        
        # Scale definitions (semitone steps)
        self.scales = {
            'major': [0, 2, 4, 5, 7, 9, 11],
            'minor': [0, 2, 3, 5, 7, 8, 10],
            'pentatonic': [0, 2, 4, 7, 9],
            'blues': [0, 3, 5, 6, 7, 10],
            'chromatic': [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11]
        }
        
        # Chord definitions (semitone offsets from root)
        self.chords = {
            'major': [0, 4, 7],
            'minor': [0, 3, 7],
            'diminished': [0, 3, 6],
            'augmented': [0, 4, 8],
            'major7': [0, 4, 7, 11],
            'minor7': [0, 3, 7, 10],
            'dominant7': [0, 4, 7, 10]
        }
    
    def _sine_wave(self, t, freq):
        """Generate a sine wave."""
        return np.sin(2 * np.pi * freq * t)
    
    def _square_wave(self, t, freq):
        """Generate a square wave."""
        return np.sign(np.sin(2 * np.pi * freq * t))
    
    def _sawtooth_wave(self, t, freq):
        """Generate a sawtooth wave."""
        return 2 * (freq * t - np.floor(0.5 + freq * t))
    
    def _triangle_wave(self, t, freq):
        """Generate a triangle wave."""
        return 2 * np.abs(2 * (freq * t - np.floor(0.5 + freq * t))) - 1
    
    def _noise(self, t, freq):
        """Generate white noise."""
        return np.random.uniform(-1, 1, len(t))
    
    def apply_envelope(self, wave, attack=0.01, decay=0.05):
        """Apply an ADSR envelope to the wave."""
        samples = len(wave)
        attack_samples = int(attack * self.sample_rate)
        decay_samples = int(decay * self.sample_rate)
        
        envelope = np.ones(samples)
        
        # Attack phase
        if attack_samples > 0:
            envelope[:attack_samples] = np.linspace(0, 1, attack_samples)
        
        # Decay phase
        if decay_samples > 0 and decay_samples < samples:
            decay_start = samples - decay_samples
            envelope[decay_start:] = np.linspace(1, 0, decay_samples)
        
        return wave * envelope
    
    def apply_filter(self, wave, cutoff, resonance):
        """Apply a simple low-pass filter."""
        # Normalize cutoff frequency
        nyquist = self.sample_rate / 2
        cutoff = min(cutoff, nyquist - 1)
        normalized_cutoff = cutoff / nyquist
        
        # Create a simple butterworth filter
        b, a = signal.butter(2, normalized_cutoff, 'low', analog=False)
        
        # Apply resonance (boost near cutoff)
        if resonance > 1.0:
            b, a = signal.butter(2, normalized_cutoff * 0.9, 'high', analog=False)
            filtered_wave = wave + (resonance - 1.0) * signal.lfilter(b, a, wave)
        else:
            filtered_wave = wave
        
        # Apply low-pass filter
        b, a = signal.butter(2, normalized_cutoff, 'low', analog=False)
        return signal.lfilter(b, a, filtered_wave)
    
    def apply_stereo(self, wave, pan):
        """Apply stereo panning to wave (0 = left, 0.5 = center, 1 = right)."""
        left_gain = min(1.0, 2.0 * (1.0 - pan))
        right_gain = min(1.0, 2.0 * pan)
        
        # Create stereo wave
        stereo_wave = np.zeros((len(wave), 2), dtype=np.int16)
        stereo_wave[:, 0] = (wave * left_gain).astype(np.int16)
        stereo_wave[:, 1] = (wave * right_gain).astype(np.int16)
        
        return stereo_wave
    
    def note_to_freq(self, note_name, octave):
        """Convert note name and octave to frequency."""
        if note_name not in self.notes:
            return 440.0  # Default to A4
        
        # Get base frequency for note
        base_freq = self.notes[note_name]
        
        # Adjust for octave (A4 = 440Hz is our reference)
        if note_name in ['A', 'A#', 'B']:
            octave_adjustment = octave - 4
        else:
            octave_adjustment = octave - 3
        
        return base_freq * (2 ** octave_adjustment)
    
    def get_scale_notes(self):
        """Get frequencies for current scale."""
        if self.scale_type not in self.scales:
            return [self.base_frequency]
        
        scale_steps = self.scales[self.scale_type]
        root_freq = self.note_to_freq(self.root_note, self.octave)
        
        return [root_freq * (2 ** (step / 12)) for step in scale_steps]
    
    def get_chord_notes(self):
        """Get frequencies for current chord."""
        if self.chord_type not in self.chords:
            return [self.base_frequency]
        
        chord_steps = self.chords[self.chord_type]
        root_freq = self.note_to_freq(self.root_note, self.octave)
        
        return [root_freq * (2 ** (step / 12)) for step in chord_steps]
    
    def create_sound(self, freq, duration=2.0):
        """Create a sound with current parameters."""
        try:
            t_vals = np.linspace(0, duration, int(self.sample_rate * duration), endpoint=False)
            wave = np.zeros_like(t_vals)
            
            # Generate waveform
            wave_func = self.waveforms.get(self.current_waveform, self._sine_wave)
            
            # Add harmonics
            for harmonic, amplitude in self.harmonics:
                wave += amplitude * self.volume * wave_func(t_vals, freq * harmonic)
            
            # Normalize wave
            if np.max(np.abs(wave)) > 0:
                wave = wave / np.max(np.abs(wave)) * self.volume
            
            # Apply envelope
            wave = self.apply_envelope(wave)
            
            # Apply filter
            try:
                if self.filter_freq > 20:  # Only filter if cutoff is above 20Hz
                    wave = self.apply_filter(wave, self.filter_freq, self.resonance)
            except:
                pass  # Skip filtering if scipy is not available or error occurs
            
            # Convert to 16-bit audio
            wave = (wave * 32767).astype(np.int16)
            
            # Apply stereo panning
            stereo_wave = self.apply_stereo(wave, self.stereo_pan)
            
            return pygame.mixer.Sound(buffer=stereo_wave)
        except Exception as e:
            print(f"Sound generation error: {e}")
            return None
    
    def play_sound(self, freq=None):
        """Play a sound with the given frequency."""
        try:
            if freq is None:
                freq = self.base_frequency
            
            # Stop previous sound
            self.stop_sound()
            
            # Create and play new sound
            sound = self.create_sound(freq)
            if sound:
                channel = sound.play(-1)  # Loop indefinitely
                self.channels[freq] = {
                    'sound': sound,
                    'channel': channel
                }
            return True
        except Exception as e:
            print(f"Sound playback error: {e}")
            return False
    
    def play_chord(self):
        """Play current chord."""
        try:
            # Stop all current sounds
            self.stop_all_sounds()
            
            # Get chord notes
            chord_freqs = self.get_chord_notes()
            
            # Play each note
            for freq in chord_freqs:
                sound = self.create_sound(freq)
                if sound:
                    # Reduce volume for chord notes
                    sound.set_volume(self.volume / len(chord_freqs))
                    channel = sound.play(-1)
                    self.channels[freq] = {
                        'sound': sound,
                        'channel': channel
                    }
            
            self.playing_chord = True
            return True
        except Exception as e:
            print(f"Chord playback error: {e}")
            return False
    
    def stop_sound(self, freq=None):
        """Stop a specific sound or the base frequency sound."""
        if freq is None:
            freq = self.base_frequency
        
        if freq in self.channels:
            channel = self.channels[freq]['channel']
            if channel:
                channel.stop()
            self.channels.pop(freq)
    
    def stop_all_sounds(self):
        """Stop all playing sounds."""
        for freq in list(self.channels.keys()):
            self.stop_sound(freq)
        self.playing_chord = False
    
    def set_waveform(self, waveform):
        """Set current waveform type."""
        if waveform in self.waveforms:
            self.current_waveform = waveform
    
    def cycle_waveform(self):
        """Cycle to next waveform type."""
        waveforms = list(self.waveforms.keys())
        current_idx = waveforms.index(self.current_waveform)
        next_idx = (current_idx + 1) % len(waveforms)
        self.current_waveform = waveforms[next_idx]
        return self.current_waveform
    
    def cycle_scale(self):
        """Cycle to next scale type."""
        scales = list(self.scales.keys())
        if self.scale_type in scales:
            current_idx = scales.index(self.scale_type)
            next_idx = (current_idx + 1) % len(scales)
            self.scale_type = scales[next_idx]
        else:
            self.scale_type = scales[0]
        return self.scale_type
    
    def cycle_chord(self):
        """Cycle to next chord type."""
        chords = list(self.chords.keys())
        if self.chord_type in chords:
            current_idx = chords.index(self.chord_type)
            next_idx = (current_idx + 1) % len(chords)
            self.chord_type = chords[next_idx]
        else:
            self.chord_type = chords[0]
        
        # Update if currently playing a chord
        if self.playing_chord:
            self.play_chord()
        
        return self.chord_type
    
    def cycle_root_note(self):
        """Cycle to next root note."""
        notes = list(self.notes.keys())
        if self.root_note in notes:
            current_idx = notes.index(self.root_note)
            next_idx = (current_idx + 1) % len(notes)
            self.root_note = notes[next_idx]
        else:
            self.root_note = notes[0]
        
        # Update base frequency
        self.base_frequency = self.note_to_freq(self.root_note, self.octave)
        
        # Update if currently playing
        if self.playing_chord:
            self.play_chord()
        
        return self.root_note, self.base_frequency

class LineWaveVisualizer:
    def __init__(self, width=1024, height=768):
        # Display settings
        self.width = width
        self.height = height
        
        # Initialize sound generator
        try:
            self.sound_generator = SoundGenerator()
            self.sound_enabled = True
        except Exception as e:
            print(f"Could not initialize sound: {e}")
            self.sound_enabled = False
        
        # Wave parameters
        self.amplitude = 0.6
        self.wave_speed = 1.0
        self.line_spacing = 0.2  # Spacing between lines
        self.num_lines = 40      # Number of horizontal and vertical lines
        self.line_width = 1.5    # Line thickness
        self.line_colors = [
            np.array([0.0, 0.8, 1.0], dtype=np.float32),  # Cyan
            np.array([0.8, 0.3, 0.8], dtype=np.float32),  # Purple
            np.array([0.0, 1.0, 0.5], dtype=np.float32),  # Green
            np.array([1.0, 0.5, 0.0], dtype=np.float32),  # Orange
            np.array([1.0, 1.0, 0.2], dtype=np.float32),  # Yellow
            np.array([0.9, 0.2, 0.3], dtype=np.float32)   # Red
        ]
        self.current_color_index = 0
        self.line_color = self.line_colors[self.current_color_index]
        
        # Camera parameters
        self.perspective_camera = np.array([0.0, 8.0, 10.0], dtype=np.float32)
        self.top_down_camera = np.array([0.0, 15.0, 0.0], dtype=np.float32)
        self.camera_position = self.perspective_camera.copy()
        self.camera_target = np.array([0.0, 0.0, 0.0], dtype=np.float32)
        self.camera_up = np.array([0.0, 1.0, 0.0], dtype=np.float32)
        self.camera_rotation_speed = 0.2
        self.auto_rotate = True
        self.top_view = False
        
        # Timing
        self.start_time = time.time()
        self.last_fps_update = 0
        self.frame_count = 0
        self.fps = 0
        
        # Create the line geometry
        self.create_lines()
        
        # Create concentric circles for top view
        self.create_circles()
        
        # Set font for overlay
        pygame.font.init()
        self.font = pygame.font.SysFont('Arial', 18)
        
        # Flags for interactivity
        self.show_info = True
        self.parallax_effect = True
        self.show_circles = True
        
        # Initialize OpenGL
        self.setup_display()
        self.init_opengl()
        
        # Start sound if enabled
        if self.sound_enabled:
            self.sound_generator.play_sound()

    def setup_display(self):
        """Set up the Pygame display."""
        self.display = (self.width, self.height)
        self.screen = pygame.display.set_mode(self.display, DOUBLEBUF | OPENGL)
        pygame.display.set_caption("Line Wave Visualizer")

    def create_lines(self):
        """Create horizontal and vertical lines for visualization."""
        # Create horizontal lines
        h_lines = []
        for i in range(self.num_lines):
            y = -np.pi + i * (2 * np.pi) / (self.num_lines - 1)
            line = []
            for j in range(100):  # 100 segments per line
                x = -np.pi + j * (2 * np.pi) / 99
                line.append([x, y])
            h_lines.append(np.array(line, dtype=np.float32))
        
        # Create vertical lines
        v_lines = []
        for i in range(self.num_lines):
            x = -np.pi + i * (2 * np.pi) / (self.num_lines - 1)
            line = []
            for j in range(100):  # 100 segments per line
                y = -np.pi + j * (2 * np.pi) / 99
                line.append([x, y])
            v_lines.append(np.array(line, dtype=np.float32))
        
        self.h_lines = h_lines
        self.v_lines = v_lines
        
        # For rendering multiple parallax layers
        self.num_layers = 5
        self.layer_offsets = [0.05 * i for i in range(self.num_layers)]

    def create_circles(self):
        """Create concentric circles for top view visualization."""
        self.circles = []
        num_circles = 10
        points_per_circle = 100
        
        for r in np.linspace(0.5, np.pi, num_circles):
            circle_points = []
            for theta in np.linspace(0, 2*np.pi, points_per_circle, endpoint=False):
                x = r * np.cos(theta)
                y = r * np.sin(theta)
                circle_points.append([x, y])
            # Close the circle
            circle_points.append(circle_points[0])
            self.circles.append(np.array(circle_points, dtype=np.float32))
        
        # Create radial lines
        self.radials = []
        num_radials = 16
        
        for theta in np.linspace(0, 2*np.pi, num_radials, endpoint=False):
            radial_points = []
            for r in np.linspace(0, np.pi, 50):
                x = r * np.cos(theta)
                y = r * np.sin(theta)
                radial_points.append([x, y])
            self.radials.append(np.array(radial_points, dtype=np.float32))

    def init_opengl(self):
        """Initialize OpenGL settings and resources."""
        # Compile and link shaders
        self.shader = compileProgram(
            compileShader(VERTEX_SHADER, GL_VERTEX_SHADER),
            compileShader(FRAGMENT_SHADER, GL_FRAGMENT_SHADER)
        )
        
        # Create VAOs and VBOs for each line
        self.h_line_vaos = []
        self.v_line_vaos = []
        
        # Setup horizontal lines
        for line in self.h_lines:
            vao = glGenVertexArrays(1)
            glBindVertexArray(vao)
            
            vbo = glGenBuffers(1)
            glBindBuffer(GL_ARRAY_BUFFER, vbo)
            glBufferData(GL_ARRAY_BUFFER, line.nbytes, line, GL_STATIC_DRAW)
            
            pos_loc = glGetAttribLocation(self.shader, "position")
            glEnableVertexAttribArray(pos_loc)
            glVertexAttribPointer(pos_loc, 2, GL_FLOAT, GL_FALSE, 0, None)
            
            glBindBuffer(GL_ARRAY_BUFFER, 0)
            glBindVertexArray(0)
            
            self.h_line_vaos.append(vao)
        
        # Setup vertical lines
        for line in self.v_lines:
            vao = glGenVertexArrays(1)
            glBindVertexArray(vao)
            
            vbo = glGenBuffers(1)
            glBindBuffer(GL_ARRAY_BUFFER, vbo)
            glBufferData(GL_ARRAY_BUFFER, line.nbytes, line, GL_STATIC_DRAW)
            
            pos_loc = glGetAttribLocation(self.shader, "position")
            glEnableVertexAttribArray(pos_loc)
            glVertexAttribPointer(pos_loc, 2, GL_FLOAT, GL_FALSE, 0, None)
            
            glBindBuffer(GL_ARRAY_BUFFER, 0)
            glBindVertexArray(0)
            
            self.v_line_vaos.append(vao)
        
        # Setup circle VAOs
        self.circle_vaos = []
        for circle in self.circles:
            vao = glGenVertexArrays(1)
            glBindVertexArray(vao)
            
            vbo = glGenBuffers(1)
            glBindBuffer(GL_ARRAY_BUFFER, vbo)
            glBufferData(GL_ARRAY_BUFFER, circle.nbytes, circle, GL_STATIC_DRAW)
            
            pos_loc = glGetAttribLocation(self.shader, "position")
            glEnableVertexAttribArray(pos_loc)
            glVertexAttribPointer(pos_loc, 2, GL_FLOAT, GL_FALSE, 0, None)
            
            glBindBuffer(GL_ARRAY_BUFFER, 0)
            glBindVertexArray(0)
            
            self.circle_vaos.append(vao)
        
        # Setup radial VAOs
        self.radial_vaos = []
        for radial in self.radials:
            vao = glGenVertexArrays(1)
            glBindVertexArray(vao)
            
            vbo = glGenBuffers(1)
            glBindBuffer(GL_ARRAY_BUFFER, vbo)
            glBufferData(GL_ARRAY_BUFFER, radial.nbytes, radial, GL_STATIC_DRAW)
            
            pos_loc = glGetAttribLocation(self.shader, "position")
            glEnableVertexAttribArray(pos_loc)
            glVertexAttribPointer(pos_loc, 2, GL_FLOAT, GL_FALSE, 0, None)
            
            glBindBuffer(GL_ARRAY_BUFFER, 0)
            glBindVertexArray(0)
            
            self.radial_vaos.append(vao)
        
        # Enable depth testing and blending
        glEnable(GL_DEPTH_TEST)
        glDepthFunc(GL_LESS)
        glEnable(GL_BLEND)
        glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA)
        
        # Initialize matrices
        self.update_matrices()

    def update_matrices(self):
        """Update model, view, and projection matrices."""
        self.model = np.eye(4, dtype=np.float32)
        self.view = lookAt(self.camera_position, self.camera_target, self.camera_up)
        self.projection = perspective(45, self.width / self.height, 0.1, 100)

    def update_camera(self, delta_time):
        """Update camera position for auto-rotation."""
        if self.auto_rotate and not self.top_view:
            # Only rotate in perspective view
            rotation = create_rotation_matrix(delta_time * self.camera_rotation_speed, np.array([0, 1, 0]))
            position = np.append(self.camera_position - self.camera_target, 1)
            rotated_position = rotation @ position
            self.camera_position = self.camera_target + rotated_position[:3]
            self.update_matrices()

    def toggle_view(self):
        """Toggle between perspective and top-down view."""
        self.top_view = not self.top_view
        if self.top_view:
            self.camera_position = self.top_down_camera.copy()
        else:
            self.camera_position = self.perspective_camera.copy()
        self.update_matrices()






    def handle_keyboard_input(self, delta_time):
        """Process keyboard input for interactive controls."""
        keys = pygame.key.get_pressed()
        
        # Frequency controls
        if self.sound_enabled:
            freq_change = 0
            if keys[K_UP]:
                freq_change = 5
            elif keys[K_DOWN]:
                freq_change = -5
            
            if freq_change != 0:
                new_freq = max(20, min(2000, self.sound_generator.base_frequency + freq_change))
                self.sound_generator.base_frequency = new_freq
                if not self.sound_generator.playing_chord:
                    self.sound_generator.play_sound()
            
            # Pan controls (left/right stereo)
            pan_change = 0
            if keys[K_LEFT]:
                pan_change = -0.02
            elif keys[K_RIGHT]:
                pan_change = 0.02
                
            if pan_change != 0:
                self.sound_generator.stereo_pan = max(0.0, min(1.0, self.sound_generator.stereo_pan + pan_change))
                if not self.sound_generator.playing_chord:
                    self.sound_generator.play_sound()
            
            # Filter controls
            filter_change = 0
            if keys[K_1]:
                filter_change = -50
            elif keys[K_2]:
                filter_change = 50
                
            if filter_change != 0:
                self.sound_generator.filter_freq = max(20, min(20000, self.sound_generator.filter_freq + filter_change))
                if not self.sound_generator.playing_chord:
                    self.sound_generator.play_sound()
            
            # Resonance controls
            resonance_change = 0
            if keys[K_3]:
                resonance_change = -0.1
            elif keys[K_4]:
                resonance_change = 0.1
                
            if resonance_change != 0:
                self.sound_generator.resonance = max(0.1, min(10.0, self.sound_generator.resonance + resonance_change))
                if not self.sound_generator.playing_chord:
                    self.sound_generator.play_sound()
        
        # Amplitude controls
        if keys[K_PAGEUP]:
            self.amplitude = min(1.5, self.amplitude + 0.02)
        elif keys[K_PAGEDOWN]:
            self.amplitude = max(0.1, self.amplitude - 0.02)
            
        # Camera controls (only in perspective view)
        if not self.top_view:
            speed = 2.0 * delta_time
            if keys[K_w]:
                self.camera_position[2] -= speed
            if keys[K_s]:
                self.camera_position[2] += speed
            if keys[K_a]:
                self.camera_position[0] -= speed
            if keys[K_d]:
                self.camera_position[0] += speed
            if keys[K_q]:
                self.camera_position[1] += speed
            if keys[K_z]:
                self.camera_position[1] -= speed
                
            if keys[K_r]:
                # Reset camera
                self.camera_position = self.perspective_camera.copy()
                self.camera_target = np.array([0.0, 0.0, 0.0], dtype=np.float32)
                
            # Update matrices if camera changed
            if keys[K_w] or keys[K_s] or keys[K_a] or keys[K_d] or keys[K_q] or keys[K_z] or keys[K_r]:
                self.perspective_camera = self.camera_position.copy()
                self.update_matrices()

    def render_frame(self, current_time, delta_time):
        """Render a single frame."""
        # Update simulation
        self.update_camera(delta_time)
        
        # Clear the frame
        glClearColor(0.02, 0.02, 0.05, 1.0)
        glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT)
        
        # Use shader program
        glUseProgram(self.shader)
        
        # Set line width
        glLineWidth(self.line_width)
        
        # Update common uniforms
        time_loc = glGetUniformLocation(self.shader, "time")
        glUniform1f(time_loc, current_time * self.wave_speed)
        
        freq_loc = glGetUniformLocation(self.shader, "frequency")
        if self.sound_enabled:
            wave_freq = (self.sound_generator.base_frequency / 440.0) * 1.5
        else:
            wave_freq = 1.5  # Default value if sound is disabled
        glUniform1f(freq_loc, wave_freq)
        
        amp_loc = glGetUniformLocation(self.shader, "amplitude")
        glUniform1f(amp_loc, self.amplitude)
        
        # Set matrices
        model_loc = glGetUniformLocation(self.shader, "model")
        glUniformMatrix4fv(model_loc, 1, GL_TRUE, self.model)
        
        view_loc = glGetUniformLocation(self.shader, "view")
        glUniformMatrix4fv(view_loc, 1, GL_TRUE, self.view)
        
        proj_loc = glGetUniformLocation(self.shader, "projection")
        glUniformMatrix4fv(proj_loc, 1, GL_TRUE, self.projection)
        
        # Set color
        color_loc = glGetUniformLocation(self.shader, "lineColor")
        
        # In top view, draw circles and radials if enabled
        if self.top_view and self.show_circles:
            # Draw concentric circles
            glUniform3fv(color_loc, 1, self.line_color * 0.8)
            offset_loc = glGetUniformLocation(self.shader, "lineOffset")
            glUniform1f(offset_loc, 0.0)
            
            for vao in self.circle_vaos:
                glBindVertexArray(vao)
                glDrawArrays(GL_LINE_STRIP, 0, 101)  # 100 points + 1 to close the circle
                
            # Draw radial lines
            for vao in self.radial_vaos:
                glBindVertexArray(vao)
                glDrawArrays(GL_LINE_STRIP, 0, 50)
        
        # Draw grid lines with parallax effect if enabled
        if self.parallax_effect:
            self.draw_with_parallax(color_loc)
        else:
            # Set base color
            glUniform3fv(color_loc, 1, self.line_color)
            offset_loc = glGetUniformLocation(self.shader, "lineOffset")
            glUniform1f(offset_loc, 0.0)
            
            # Draw horizontal lines
            for vao in self.h_line_vaos:
                glBindVertexArray(vao)
                glDrawArrays(GL_LINE_STRIP, 0, 100)
            
            # Draw vertical lines
            for vao in self.v_line_vaos:
                glBindVertexArray(vao)
                glDrawArrays(GL_LINE_STRIP, 0, 100)
        
        # Unbind vertex array
        glBindVertexArray(0)
        
        # Render overlay information
        if self.show_info:
            self.render_overlay()

    def draw_with_parallax(self, color_loc):
        """Draw lines with parallax effect using multiple offset layers."""
        offset_loc = glGetUniformLocation(self.shader, "lineOffset")
        
        for i, offset in enumerate(self.layer_offsets):
            # Adjust color for depth - deeper layers are more transparent
            alpha = 1.0 - (i / self.num_layers) * 0.7
            layer_color = self.line_color * alpha
            glUniform3fv(color_loc, 1, layer_color)
            
            # Set offset for this layer
            glUniform1f(offset_loc, offset)
            
            # Draw horizontal lines for this layer
            for vao in self.h_line_vaos:
                glBindVertexArray(vao)
                glDrawArrays(GL_LINE_STRIP, 0, 100)
            
            # Draw vertical lines for this layer
            for vao in self.v_line_vaos:
                glBindVertexArray(vao)
                glDrawArrays(GL_LINE_STRIP, 0, 100)

    def render_overlay(self):
        """Render text overlay with information."""
        # Switch to 2D rendering for text
        glUseProgram(0)
        glMatrixMode(GL_PROJECTION)
        glPushMatrix()
        glLoadIdentity()
        glOrtho(0, self.width, self.height, 0, -1, 1)
        glMatrixMode(GL_MODELVIEW)
        glPushMatrix()
        glLoadIdentity()
        glDisable(GL_DEPTH_TEST)
        
        # Draw text overlay
        self.draw_text(10, 10, f"FPS: {self.fps:.1f}")
        
        if self.sound_enabled:
            self.draw_text(10, 30, f"Frequency: {self.sound_generator.base_frequency:.1f} Hz")
            self.draw_text(10, 50, f"Waveform: {self.sound_generator.current_waveform}")
            self.draw_text(10, 70, f"Pan: {self.sound_generator.stereo_pan:.2f}")
            self.draw_text(10, 90, f"Filter: {self.sound_generator.filter_freq:.0f} Hz")
            self.draw_text(10, 110, f"Resonance: {self.sound_generator.resonance:.1f}")
            if self.sound_generator.playing_chord:
                self.draw_text(10, 130, f"Chord: {self.sound_generator.root_note} {self.sound_generator.chord_type}")
            else:
                self.draw_text(10, 130, f"Note: {self.sound_generator.root_note}{self.sound_generator.octave}")
        
        self.draw_text(10, 150, f"View: {'Top-Down' if self.top_view else 'Perspective'}")
        self.draw_text(10, 170, f"Amplitude: {self.amplitude:.2f}")
        
        # Controls help
        y_pos = self.height - 180
        self.draw_text(10, y_pos, "Sound Controls:")
        self.draw_text(10, y_pos + 20, "↑/↓: Frequency, ←/→: Pan left/right")
        self.draw_text(10, y_pos + 40, "1/2: Filter cutoff, 3/4: Resonance")
        self.draw_text(10, y_pos + 60, "W: Change waveform, N: Change note, C: Chord/note toggle")
        
        self.draw_text(10, y_pos + 90, "View Controls:")
        self.draw_text(10, y_pos + 110, "T: Toggle top-down view, P: Toggle parallax")
        self.draw_text(10, y_pos + 130, "O: Toggle circles (top view), Space: Toggle rotation")
        self.draw_text(10, y_pos + 150, "I: Toggle info, L: Cycle colors, M: Toggle sound, Esc: Quit")
        
        # Restore 3D rendering
        glEnable(GL_DEPTH_TEST)
        glMatrixMode(GL_PROJECTION)
        glPopMatrix()
        glMatrixMode(GL_MODELVIEW)
        glPopMatrix()

    def draw_text(self, x, y, text, color=(255, 255, 255)):
        """Draw text on the screen."""
        try:
            text_surface = self.font.render(text, True, color)
            text_data = pygame.image.tostring(text_surface, "RGBA", True)
            width, height = text_surface.get_size()
            
            glEnable(GL_BLEND)
            glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA)
            glRasterPos2d(x, y)
            glDrawPixels(width, height, GL_RGBA, GL_UNSIGNED_BYTE, text_data)
            glDisable(GL_BLEND)
        except Exception as e:
            print(f"Text rendering error: {e}")

    def update_fps(self, current_time):
        """Update FPS counter."""
        self.frame_count += 1
        if current_time - self.last_fps_update >= 1.0:  # Update every second
            self.fps = self.frame_count / (current_time - self.last_fps_update)
            self.frame_count = 0
            self.last_fps_update = current_time

    def run(self):
        """Main application loop."""
        clock = pygame.time.Clock()
        running = True
        last_time = time.time()
        
        # Main loop
        while running:
            current_time = time.time() - self.start_time
            delta_time = time.time() - last_time
            last_time = time.time()
            
            # Process events
            for event in pygame.event.get():
                if event.type == QUIT:
                    running = False
                elif event.type == KEYDOWN:
                    if event.key == K_ESCAPE:
                        running = False
                    elif event.key == K_i:
                        self.show_info = not self.show_info
                    elif event.key == K_t:
                        self.toggle_view()
                    elif event.key == K_p:
                        self.parallax_effect = not self.parallax_effect
                    elif event.key == K_o:
                        self.show_circles = not self.show_circles
                    elif event.key == K_SPACE:
                        self.auto_rotate = not self.auto_rotate
                    elif event.key == K_l:
                        # Cycle through color presets
                        self.current_color_index = (self.current_color_index + 1) % len(self.line_colors)
                        self.line_color = self.line_colors[self.current_color_index]
                    
                    # Sound-related events
                    if self.sound_enabled:
                        if event.key == K_m:
                            # Toggle sound
                            if self.sound_generator.channels:
                                self.sound_generator.stop_all_sounds()
                            else:
                                self.sound_generator.play_sound()
                        elif event.key == K_w:
                            # Cycle waveform
                            waveform = self.sound_generator.cycle_waveform()
                            print(f"Waveform: {waveform}")
                            if not self.sound_generator.playing_chord:
                                self.sound_generator.play_sound()
                        elif event.key == K_n:
                            # Cycle root note
                            note, freq = self.sound_generator.cycle_root_note()
                            print(f"Root note: {note}, Frequency: {freq}")
                            if not self.sound_generator.playing_chord:
                                self.sound_generator.play_sound()
                        elif event.key == K_c:
                            # Toggle between chord and single note
                            if self.sound_generator.playing_chord:
                                self.sound_generator.stop_all_sounds()
                                self.sound_generator.play_sound()
                                self.sound_generator.playing_chord = False
                                print(f"Playing note: {self.sound_generator.root_note}")
                            else:
                                self.sound_generator.play_chord()
                                print(f"Playing chord: {self.sound_generator.root_note} {self.sound_generator.chord_type}")
                        elif event.key == K_v:
                            # Cycle chord type
                            chord_type = self.sound_generator.cycle_chord()
                            print(f"Chord type: {chord_type}")
                            if self.sound_generator.playing_chord:
                                self.sound_generator.play_chord()
            
            # Process continuous keyboard input
            self.handle_keyboard_input(delta_time)
            
            # Render frame
            self.render_frame(current_time, delta_time)
            
            # Update FPS counter
            self.update_fps(current_time)
            
            # Display the rendered frame
            pygame.display.flip()
            
            # Cap the frame rate
            clock.tick(60)
        
        # Clean up
        if self.sound_enabled:
            self.sound_generator.stop_all_sounds()
        pygame.quit()


if __name__ == "__main__":
    pygame.init()
    try:
        import scipy.signal
        has_scipy = True
    except ImportError:
        print("Warning: scipy not found, some audio features will be disabled")
        has_scipy = False
        
    visualizer = LineWaveVisualizer(width=1024, height=768)
    visualizer.run()
