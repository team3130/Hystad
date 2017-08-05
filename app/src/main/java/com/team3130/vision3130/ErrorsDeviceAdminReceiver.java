package com.team3130.vision3130;

import android.app.admin.DeviceAdminReceiver;
import android.content.ComponentName;
import android.content.Context;

public class ErrorsDeviceAdminReceiver extends DeviceAdminReceiver { //changed from ChezyDeviceAdminReciever to ErrorsDeviceAdminReceiver

    public static ComponentName getComponentName(Context context) {
        return new ComponentName(context.getApplicationContext(), ErrorsDeviceAdminReceiver.class);
    }

}