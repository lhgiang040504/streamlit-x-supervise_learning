            accuracy_values = [F1Train, F1Test]
            accuracy_values = np.array([round(accuracy_value*100, 2) for accuracy_value in accuracy_values ])
            cm = metrics.confusion_matrix(y_test, y_pred)
            
            # Show biểu đồ cột
            st.title(" :violet[Drawexplicitly chart]")
            labels = np.array(['F1 Score Train', 'F1 Score Test'])
            fig, ax = plt.subplots()
            ax.bar(labels, accuracy_values, 0.6, 0.01)
            ax.set_xticks(labels)
            ax.set_yticks(range(0, 101, 10))
            plt.xlabel(algorithm)
            plt.ylabel('F1 Score (%)')
            for ind,val in enumerate(accuracy_values):
                plt.text(ind, val + 0.6, str(val), transform = plt.gca().transData,horizontalalignment = 'center', color = 'red',fontsize = 'small')            
            
            cm = confusion_matrix(y_test, y_pred, labels = model.classes_)
            disp = ConfusionMatrixDisplay(confusion_matrix = cm, display_labels = model.classes_)
            disp.plot()
            st.pyplot()

            F1Train = metrics.f1_score(y_pred, y_train, average = 'macro')
            F1Test = metrics.f1_score(y_pred, y_test, average = 'macro')