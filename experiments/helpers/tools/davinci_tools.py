import os
import logging

from .python_get_resolve import GetResolve 
from langchain_core.tools import tool
from typing import Dict, List

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
    
class ResolveProject:
    def __init__(self):
        self.resolve = GetResolve()
        self.project_manager = None
        self.project = None
        self.media_pool = None
        self.timeline = None
        self.logger = logging.getLogger(__name__)
        
        try:
            if self.resolve is None:
                raise Exception("DaVinci Resolve is not running. DaVinci Resolve must be running to use this tool.")
            self.project_manager = self.resolve.GetProjectManager()
        except Exception as e:
            self.logger.error(f"An unexpected error occurred: {e}")
            raise

    def create_project(self, project_name: str, video_settings: Dict[str, int]) -> bool:
        try:
            existing_project = self.project_manager.LoadProject(project_name)
            if existing_project:
                self.logger.info(f"Project '{project_name}' already exists. Loading existing project.")
                self.project = existing_project
            else:
                self.project = self.project_manager.CreateProject(project_name)
                if not self.project:
                    raise Exception(f"Failed to create project: {project_name}")
                self.logger.info(f"Created new project: {project_name}")

            self.media_pool = self.project.GetMediaPool()
            root_folder = self.media_pool.GetRootFolder()

            # Apply video settings
            self.project.SetSetting("timelineFrameRate", str(video_settings.get("frame_rate", 24)))
            self.project.SetSetting("timelineResolutionWidth", str(video_settings.get("width", 1920)))
            self.project.SetSetting("timelineResolutionHeight", str(video_settings.get("height", 1080)))

            timeline_count = self.project.GetTimelineCount()
            if timeline_count > 0:
                self.logger.info(f"Project already has {timeline_count} timeline(s). Using the first timeline.")
                self.timeline = self.project.GetTimelineByIndex(1)
            else:
                self.logger.info("No existing timeline found. Creating a new timeline.")
                self.timeline = self.media_pool.CreateEmptyTimeline("Main Timeline")
                if not self.timeline:
                    raise Exception("Failed to create timeline")
            
            self.project.SetCurrentTimeline(self.timeline)
            
            # Set timeline length
            if "length" in video_settings:
                end_frame = int(video_settings["length"] * video_settings.get("frame_rate", 24))
                self.timeline.SetSetting("timelineOutPoint", str(end_frame))

            return True
        except Exception as e:
            self.logger.error(f"Error creating project: {str(e)}")
            return False

    def add_audio_track(self, audio_file_path: str) -> bool:
        try:
            if not self.timeline:
                raise Exception("Timeline not initialized")
            
            if not os.path.exists(audio_file_path):
                raise Exception(f"Audio file not found: {audio_file_path}")
            
            # Add new audio track
            # current_track_count = self.timeline.GetTrackCount("audio")
            # success = self.timeline.AddTrack("audio", "stereo")
            # if not success:
            #     raise Exception("Failed to add audio track")
            
            # new_track_index = current_track_count - 1
                        
            # Import audio file
            imported_audio = self.media_pool.ImportMedia(audio_file_path)
            if not imported_audio:
                raise Exception(f"Failed to import audio file: {audio_file_path}")
            
            # Add audio to the timeline
            audio_clip = imported_audio[0]  # Assuming ImportMedia returns a list
            audio_item = self.media_pool.AppendToTimeline(imported_audio)
            if not audio_item:
                raise Exception("Failed to add audio to the timeline")
            
            
            self.logger.info(f"Added new audio track with file: {audio_file_path}")
            return True
        except Exception as e:
            self.logger.error(f"Error adding audio track: {str(e)}")
            return False


    def add_video_clips(self, clip_paths: List[str]) -> List[str]:
        try:
            if not self.media_pool or not self.timeline:
                raise Exception("Media pool or timeline not initialized")
            
            added_clips = []
            for clip_path in clip_paths:
                if not os.path.exists(clip_path):
                    self.logger.warning(f"Clip not found: {clip_path}")
                    continue
                
                imported_clips = self.media_pool.ImportMedia(clip_path)
                if not imported_clips:
                    self.logger.warning(f"Failed to import clip: {clip_path}")
                    continue
                
                for clip in imported_clips:
                    success = self.timeline.AppendToTimeline(clip)
                    if success:
                        added_clips.append(clip_path)
                        self.logger.info(f"Added clip to timeline: {clip_path}")
                    else:
                        self.logger.warning(f"Failed to add clip to timeline: {clip_path}")
            
            return added_clips
        except Exception as e:
            self.logger.error(f"Error adding video clips: {str(e)}")
            return []

resolve_project = ResolveProject()

@tool
def create_resolve_project(project_name: str, length: float, width: int, height: int, frame_rate: int) -> bool:
    """
    Creates a new DaVinci Resolve project with the given name and video settings.
    
    Args:
    project_name (str): The name of the project to create or load.
    length (float): The length of the video in seconds.
    width (int): The width of the video in pixels.
    height (int): The height of the video in pixels.
    frame_rate (int): The frame rate of the video.
    
    Returns:
    bool: True if the project was created successfully, False otherwise.
    """
    video_settings = {
        "length": length,
        "width": width,
        "height": height,
        "frame_rate": frame_rate
    }
    success = resolve_project.create_project(project_name, video_settings)
    return success

@tool
def add_audio_track(audio_file_path: str) -> bool:
    """
    Adds a new audio track to the current DaVinci Resolve project and imports the specified audio file.
    
    Args:
    audio_file_path (str): The file path of the audio file to be added to the new track.
    
    Returns:
    bool: True if the audio track was added successfully, False otherwise.
    """
    success = resolve_project.add_audio_track(audio_file_path)
    return success

@tool
def add_video_clips(clip_paths: List[str]) -> str:
    """
    Adds video clips to the timeline of the current DaVinci Resolve project.
    
    Args:
    clip_paths (List[str]): A list of file paths to the video clips.
    
    Returns:
    str: A message indicating the result of the operation.
    """
    added_clips = resolve_project.add_video_clips(clip_paths)
    return len(added_clips) == len(clip_paths)

if __name__ == "__main__":
    resolve_project.create_project("My New Project", {
        "length": 300,
        "width": 1920,
        "height": 1080,
        "frame_rate": 30
    })

    resolve_project.add_audio_track("D:/source/vscode/ai_workbench/output.mp3")