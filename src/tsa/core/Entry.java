/*
 * To change this template, choose Tools | Templates
 * and open the template in the editor.
 */
package tsa.core;

import java.util.HashMap;
import java.util.Map;

/**
 *
 * @author felipe
 * Represents a textual entry which will be used as training example
 */
public class Entry {

    private String content; // main content of the entry
    private boolean valid; // to check weather the Entry was parsed correctly
    
    public Map<String, Object> features; // features and their values
    public Map<String, Object> metaData; // possible metadata 
    

    public Entry() {
        this.features = new HashMap<String, Object>();
        this.metaData = new HashMap<String, Object>();
        this.valid=false;
    }

    public String getContent() {
        return content;
    }

    public void setContent(String content) {
        this.content = content;
    }

    public Map<String, Object> getFeatures() {
        return features;
    }

    public void setFeatures(Map<String, Object> features) {
        this.features = features;
    }

    public Map<String, Object> getMetaData() {
        return metaData;
    }

    public void setMetaData(Map<String, Object> metaData) {
        this.metaData = metaData;
    }


    public boolean isValid() {
        return valid;
    }

    public void setValid(boolean valid) {
        this.valid = valid;
    }
    
    
    
    
    

    @Override
    public String toString() {
        String value = content;
        for (String metaDat : this.getMetaData().keySet()) {
            String datValue = this.getMetaData().get(metaDat).toString();
            value += "\n" + metaDat + ":" + datValue;
        }
        for (String feat : this.getFeatures().keySet()) {
            String datValue = this.getFeatures().get(feat).toString();
            value += "\n" + feat + ":" + datValue;
        }



        return value;
    }
}
