package com.team3130.vision3130.comm.messages;

import com.team3130.vision3130.comm.VisionUpdate;

public class TargetUpdateMessage extends VisionMessage {

    VisionUpdate mUpdate;
    long mTimestamp;

    public TargetUpdateMessage(VisionUpdate update, long timestamp) {
        mUpdate = update;
        mTimestamp = timestamp;
    }
    @Override
    public String getType() {
        return "targets";
    }

    @Override
    public String getMessage() {
        return mUpdate.getSendableJsonString(mTimestamp);
    }
}
//Todo: make a new type of VisionMessage that contains debug data that can be sent back to roboRIO
// Maybe use the heartbeat message to do this.
